#!/usr/bin/env python3
"""
DICOM Study to CSV+JPG Converter
Converts one .tgz study to CSV (metadata) + JPG images (pixel arrays)

Usage:
    python dicom_to_csv_jpg.py /path/to/study.tgz

Configuration:
    Edit the variables at the top of this script to set your paths
"""

import os
import sys
import tarfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

import tempfile as _tf

# Overridable by pipeline (monkeypatch before DicomStudyProcessor())
TEMP_BASE_DIR = os.environ.get(
    "VIDEO_CHAT_DICOM_TEMP", os.path.join(_tf.gettempdir(), "video_chat_ui_dicom_temp")
)
OUTPUT_BASE_DIR = os.environ.get("VIDEO_CHAT_DICOM_OUT", os.path.join(_tf.gettempdir(), "video_chat_ui_dicom_out"))

# Whether to keep intermediate files for debugging
DEBUG_KEEP_TEMP = False

# JPG quality (1-100, higher is better quality but larger files)
JPG_QUALITY = 95

# Video creation settings
CREATE_MP4_FOR_MULTIFRAME = True  # Convert multi-frame folders to MP4
MP4_FPS = 30  # Frames per second for MP4 videos
# Codec order: mp4v is what actually works (FFmpeg falls back to this anyway), then others as fallbacks
MP4_CODECS = ['mp4v', 'XVID', 'H264', 'MJPG']  # Try codecs in order of what actually works
DELETE_JPGS_AFTER_MP4 = True  # Delete JPG files after successful MP4 creation

# ============================================================================

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    pydicom = None  # type: ignore

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore


def _require_preprocessing_deps():
    if not DICOM_AVAILABLE or not PIL_AVAILABLE:
        raise ImportError(
            "DICOM preprocessing requires: pip install 'video-chat-ui[preprocessing]'"
        )
    if CREATE_MP4_FOR_MULTIFRAME and not CV2_AVAILABLE:
        raise ImportError(
            "OpenCV required for MP4: pip install 'video-chat-ui[preprocessing]'"
        )

class DicomStudyProcessor:
    def __init__(self, tgz_path):
        _require_preprocessing_deps()
        self.tgz_path = Path(tgz_path)
        self.temp_dir = None
        self.output_dir = None
        self.images_dir = None
        self.study_name = self.tgz_path.stem.replace('.tar', '').replace(".tgz", "")

        if not self.tgz_path.exists():
            raise FileNotFoundError(f"TGZ file not found: {self.tgz_path}")

        self._setup_directories()
        
        # Data storage
        self.metadata_list = []
        self.detected_codec = None  # Store the working codec for this session
        self.study_failed = False  # Track if study should be considered failed due to corruption
        self.stats = {
            'total_dicoms': 0,
            'processed_dicoms': 0,
            'failed_dicoms': 0,
            'total_frames_saved': 0,
            'mp4_videos_created': 0,
            'jpgs_converted_to_mp4': 0,
            'unique_metadata_keys': set(),
            'errors': []
        }

    # ===================== NEW HELPERS FOR JSON-SERIALIZABLE METADATA =====================

    def _tag_name(self, elem):
        """
        Prefer DICOM keyword; fall back to (gggg,eeee) tag.
        """
        try:
            if elem.keyword and elem.keyword != '':
                return elem.keyword
        except Exception:
            pass
        try:
            return f"({int(elem.tag.group):04X},{int(elem.tag.element):04X})"
        except Exception:
            return str(getattr(elem, "tag", "UNKNOWN_TAG"))

    def _element_to_primitive(self, elem):
        """
        Convert a pydicom DataElement to JSON-serializable primitive(s).
        Handles sequences (SQ), nested Datasets, MultiValue, PersonName, numbers, numpy, bytes, etc.
        """
        value = elem.value
        return self._to_primitive(value)

    def _to_primitive(self, value):
        """
        Convert arbitrary pydicom values into JSON-serializable primitives:
        dict/list/str/float/int/bool/None. Nested Datasets and Sequences become dicts/lists.
        """
        import pydicom
        from pydicom.multival import MultiValue
        from pydicom.valuerep import PersonName, PersonNameBase, DSfloat, DSdecimal, IS

        # Dataset -> dict of elements
        if isinstance(value, pydicom.dataset.Dataset):
            out = {}
            for sub in value:
                if sub.keyword == 'PixelData':
                    continue
                key = self._tag_name(sub)
                out[key] = self._element_to_primitive(sub)
            return out

        # Sequence (list of Datasets)
        if isinstance(value, list) and all(isinstance(v, pydicom.dataset.Dataset) for v in value):
            return [self._to_primitive(v) for v in value]

        # MultiValue -> list
        if isinstance(value, MultiValue):
            return [self._to_primitive(v) for v in value]

        # Known wrapper types -> native types / strings
        if isinstance(value, (PersonName, PersonNameBase)):
            return str(value)
        if isinstance(value, (DSfloat, DSdecimal)):
            try:
                return float(value)
            except Exception:
                return str(value)
        if isinstance(value, IS):
            try:
                return int(value)
            except Exception:
                # sometimes IS wraps non-int text; fall back to string
                return str(value)

        # bytes -> try decode; else length descriptor
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode('utf-8')
            except Exception:
                return f"<{len(value)} bytes>"

        # numpy scalars/arrays -> python scalars / lists
        try:
            import numpy as np
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                # Avoid blowing up CSV with giant arrays except sequences; but still serialize
                return value.tolist()
        except Exception:
            pass

        # Everything else: int/float/str/bool/None
        return value

    # =======================================================================================

    def _setup_directories(self):
        """Setup temporary and output directories"""
        # Create temporary directory
        temp_base = Path(TEMP_BASE_DIR)
        temp_base.mkdir(parents=True, exist_ok=True)
        self.temp_dir = temp_base / f"temp_{self.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory
        output_base = Path(OUTPUT_BASE_DIR)
        output_base.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_base / self.study_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images directory within the study directory
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Temp directory: {self.temp_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📁 Images directory: {self.images_dir}")
    
    def extract_study(self):
        """Extract the .tgz file to temporary directory"""
        print(f"📦 Extracting {self.tgz_path.name}...")
        
        try:
            with tarfile.open(self.tgz_path, 'r:gz') as tar:
                tar.extractall(path=self.temp_dir)
            print(f"✅ Extracted to {self.temp_dir}")
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def find_dicom_files(self):
        """Find all DICOM files in the extracted directory"""
        dicom_files = []
        
        # Common DICOM file patterns
        patterns = ['*.dcm', '*.dicom', '*.DCM', '*.DICOM']
        
        for pattern in patterns:
            dicom_files.extend(self.temp_dir.rglob(pattern))
        
        # Also check files without extension that might be DICOM
        for file_path in self.temp_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix == '':
                if self._is_dicom_file(file_path):
                    dicom_files.append(file_path)
        
        # Remove duplicates and sort
        dicom_files = sorted(list(set(dicom_files)))
        
        print(f"📋 Found {len(dicom_files)} DICOM files")
        self.stats['total_dicoms'] = len(dicom_files)
        
        return dicom_files
    
    def _is_dicom_file(self, file_path):
        """Check if a file is a DICOM file"""
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except:
            return False
    
    def _normalize_pixel_array(self, pixel_array):
        """Normalize pixel array to 8-bit for JPG saving"""
        try:
            # Convert to float for processing
            array = pixel_array.astype(np.float64)
            
            # Handle different bit depths and rescaling
            # Normalize to 0-255 range
            array_min = np.min(array)
            array_max = np.max(array)
            
            if array_max > array_min:
                # Scale to 0-255
                array = (array - array_min) / (array_max - array_min) * 255.0
            else:
                # Handle case where all pixels have the same value
                if array.ndim >= 3 and array.shape[-1] == 3:
                    # RGB image - set to middle gray for all channels
                    array = np.full_like(array, 128.0)
                else:
                    # Grayscale - set to middle gray
                    array = np.full_like(array, 128.0)
            
            # Convert to 8-bit unsigned integer
            return array.astype(np.uint8)
            
        except Exception as e:
            print(f"      ⚠️  Error normalizing pixel array: {e}")
            # Return a default image if normalization fails
            if pixel_array.ndim >= 3 and pixel_array.shape[-1] == 3:
                # RGB default
                return np.full_like(pixel_array, 128, dtype=np.uint8)
            else:
                # Grayscale default
                return np.full_like(pixel_array, 128, dtype=np.uint8)
    
    def _save_pixel_array_as_jpg(self, pixel_array, dicom_index, dicom_filename):
        """Save pixel array as JPG image(s) in appropriate directory structure"""
        try:
            # Create directory for this DICOM
            dicom_folder = self.images_dir / f"dicom_{dicom_index:04d}"
            dicom_folder.mkdir(exist_ok=True)
            
            frames_saved = 0
            
            # Determine if this is RGB or grayscale based on array shape
            is_rgb = pixel_array.ndim >= 3 and pixel_array.shape[-1] == 3
            
            # Handle different array dimensions
            if pixel_array.ndim == 2:
                # Single frame grayscale (2D array)
                normalized_array = self._normalize_pixel_array(pixel_array)
                image = Image.fromarray(normalized_array, mode='L')  # 'L' for grayscale
                
                jpg_path = dicom_folder / "frame_000.jpg"
                image.save(jpg_path, 'JPEG', quality=JPG_QUALITY)
                frames_saved = 1
                
            elif pixel_array.ndim == 3:
                if pixel_array.shape[-1] == 3:
                    # Single frame RGB (H, W, 3)
                    normalized_array = self._normalize_pixel_array(pixel_array)
                    image = Image.fromarray(normalized_array, mode='RGB')
                    
                    jpg_path = dicom_folder / "frame_000.jpg"
                    image.save(jpg_path, 'JPEG', quality=JPG_QUALITY)
                    frames_saved = 1
                else:
                    # Multi-frame grayscale (frames, H, W)
                    num_frames = pixel_array.shape[0]
                    
                    for frame_idx in range(num_frames):
                        frame = pixel_array[frame_idx]
                        normalized_frame = self._normalize_pixel_array(frame)
                        image = Image.fromarray(normalized_frame, mode='L')
                        
                        jpg_path = dicom_folder / f"frame_{frame_idx:03d}.jpg"
                        image.save(jpg_path, 'JPEG', quality=JPG_QUALITY)
                        frames_saved += 1
                    
            elif pixel_array.ndim == 4:
                if pixel_array.shape[-1] == 3:
                    # Multi-frame RGB (frames, H, W, 3) - this matches your (153, 600, 800, 3)
                    num_frames = pixel_array.shape[0]
                    
                    for frame_idx in range(num_frames):
                        frame = pixel_array[frame_idx]  # Shape: (600, 800, 3)
                        normalized_frame = self._normalize_pixel_array(frame)
                        image = Image.fromarray(normalized_frame, mode='RGB')
                        
                        jpg_path = dicom_folder / f"frame_{frame_idx:03d}.jpg"
                        image.save(jpg_path, 'JPEG', quality=JPG_QUALITY)
                        frames_saved += 1
                else:
                    # 4D grayscale - handle as frames in first dimension
                    num_frames = pixel_array.shape[0]
                    
                    for frame_idx in range(num_frames):
                        frame = pixel_array[frame_idx]
                        # If the frame is still 3D, take the middle slice or average
                        if frame.ndim == 3:
                            frame = np.mean(frame, axis=2)
                        
                        normalized_frame = self._normalize_pixel_array(frame)
                        image = Image.fromarray(normalized_frame, mode='L')
                        
                        jpg_path = dicom_folder / f"frame_{frame_idx:03d}.jpg"
                        image.save(jpg_path, 'JPEG', quality=JPG_QUALITY)
                        frames_saved += 1
            
            else:
                print(f"      ⚠️  Unsupported pixel array dimensions: {pixel_array.shape}")
                return 0
            
            self.stats['total_frames_saved'] += frames_saved
            return frames_saved
            
        except Exception as e:
            print(f"      ❌ Error saving JPG for {dicom_filename}: {e}")
            return 0
    
    def process_dicom_file(self, dicom_path, index):
        """Process a single DICOM file and extract metadata + save pixel data as JPG"""
        try:
            # Read DICOM file
            ds = pydicom.dcmread(dicom_path)
            
            # Extract metadata (everything except PixelData)
            metadata = {}
            metadata['dicom_index'] = index  # Link to image folder
            metadata['dicom_filename'] = dicom_path.name
            metadata['dicom_path_relative'] = str(dicom_path.relative_to(self.temp_dir))
            
            # Extract all DICOM metadata with JSON-safe serialization
            for elem in ds:
                if elem.keyword == 'PixelData':
                    continue  # Skip pixel data
                key = self._tag_name(elem)
                try:
                    value = self._element_to_primitive(elem)
                    # Store as JSON strings when dict/list, otherwise raw value
                    if isinstance(value, (dict, list)):
                        metadata[key] = json.dumps(value, ensure_ascii=False, default=str)
                    else:
                        metadata[key] = value
                    # Track unique keys
                    self.stats['unique_metadata_keys'].add(key)
                except Exception:
                    # Fallback to string
                    metadata[key] = str(elem.value)
                    self.stats['unique_metadata_keys'].add(key)
            
            # Extract and save pixel data if available
            frames_saved = 0
            if hasattr(ds, 'pixel_array'):
                try:
                    pixel_array = ds.pixel_array
                    # Add shape information to metadata
                    metadata['pixel_shape'] = str(pixel_array.shape)
                    metadata['pixel_dtype'] = str(pixel_array.dtype)
                    
                    # Determine if RGB or grayscale
                    is_rgb = pixel_array.ndim >= 3 and pixel_array.shape[-1] == 3
                    metadata['image_type'] = 'RGB' if is_rgb else 'GRAYSCALE'
                    
                    # Save as JPG images
                    frames_saved = self._save_pixel_array_as_jpg(pixel_array, index, dicom_path.name)
                    metadata['frames_saved'] = frames_saved
                    metadata['image_folder'] = f"images/dicom_{index:04d}"
                    
                except Exception as e:
                    print(f"      ⚠️  Could not extract/save pixel data: {e}")
                    metadata['pixel_shape'] = 'ERROR'
                    metadata['pixel_dtype'] = 'ERROR'
                    metadata['image_type'] = 'ERROR'
                    metadata['frames_saved'] = 0
                    metadata['image_folder'] = 'ERROR'
            else:
                metadata['pixel_shape'] = 'NO_PIXEL_DATA'
                metadata['pixel_dtype'] = 'NO_PIXEL_DATA'
                metadata['image_type'] = 'NO_PIXEL_DATA'
                metadata['frames_saved'] = 0
                metadata['image_folder'] = 'NO_PIXEL_DATA'
            
            # Store metadata
            self.metadata_list.append(metadata)
            
            # Initialize MP4-related metadata (will be updated later if MP4 is created)
            metadata['has_mp4'] = False
            metadata['mp4_filename'] = 'NOT_PROCESSED'
            metadata['individual_jpgs'] = True
            
            # Print progress
            shape_str = metadata.get('pixel_shape', 'N/A')
            series_desc = metadata.get('SeriesDescription', 'Unknown')
            image_type = metadata.get('image_type', 'N/A')
            frames_str = f"({frames_saved} {image_type.lower()} frames)" if frames_saved > 0 else "(no images)"
            print(f"   ✅ [{index:3d}] {dicom_path.name[:35]:<35} | {shape_str:<20} | {frames_str:<20} | {series_desc}")
            
            self.stats['processed_dicoms'] += 1
            return True
            
        except Exception as e:
            error_msg = f"Error processing {dicom_path.name}: {str(e)}"
            
            # Check if this is a corruption error that should fail the entire study
            if "fewer fragments than frames" in str(e) or "dataset may be corrupt" in str(e):
                print(f"   ❌ [{index:3d}] CORRUPTION DETECTED: {dicom_path.name}")
                print(f"        {str(e)}")
                print(f"   🚫 STUDY MARKED AS FAILED - Corruption detected")
                self.study_failed = True
                self.stats['errors'].append(f"CORRUPTION: {error_msg}")
            else:
                print(f"   ❌ [{index:3d}] {error_msg}")
                self.stats['errors'].append(error_msg)
            
            self.stats['failed_dicoms'] += 1
            return False
    
    def save_csv_file(self):
        """Save metadata to CSV file"""
        csv_path = self.output_dir / f"{self.study_name}_metadata.csv"
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.metadata_list)
            
            # Ensure any remaining dict/list values are JSON-encoded
            for col in df.columns:
                if df[col].map(lambda v: isinstance(v, (dict, list))).any():
                    df[col] = df[col].apply(lambda v: json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v)

            # Sort by dicom_index
            if 'dicom_index' in df.columns:
                df = df.sort_values('dicom_index')
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            print(f"📊 CSV saved: {csv_path}")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Unique metadata keys: {len(self.stats['unique_metadata_keys'])}")
            
            return csv_path
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return None
    
    def _create_mp4_from_jpgs(self, dicom_folder, jpg_files):
        """Create MP4 video from JPG files in a folder"""
        try:
            if not CV2_AVAILABLE:
                print(f"      ⚠️  OpenCV not available, skipping MP4 creation for {dicom_folder.name}")
                return False
            
            # Sort JPG files by frame number
            jpg_files = sorted(jpg_files, key=lambda x: int(x.stem.split('_')[1]))
            
            if len(jpg_files) < 2:
                return False  # Don't create MP4 for single frames
            
            # Read first image to get dimensions and format
            first_img = cv2.imread(str(jpg_files[0]))
            if first_img is None:
                print(f"      ⚠️  Could not read first image: {jpg_files[0]}")
                return False
            
            height, width = first_img.shape[:2]
            
            # Ensure dimensions are even (required by some codecs)
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1
            
            # Create MP4 file path
            mp4_path = dicom_folder / f"{dicom_folder.name}.mp4"
            
            # Try different codecs until one works (only if not already detected)
            video_writer = None
            used_codec = None
            
            if self.detected_codec:
                # Use previously detected working codec
                try:
                    fourcc = cv2.VideoWriter_fourcc(*self.detected_codec)
                    video_writer = cv2.VideoWriter(str(mp4_path), fourcc, MP4_FPS, (width, height))
                    if video_writer.isOpened():
                        used_codec = self.detected_codec
                except Exception:
                    # Fall back to codec detection if previously detected codec fails
                    self.detected_codec = None
            
            if video_writer is None:
                # Detect working codec
                for codec in MP4_CODECS:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        temp_writer = cv2.VideoWriter(str(mp4_path), fourcc, MP4_FPS, (width, height))
                        
                        if temp_writer.isOpened():
                            # Test write a frame to ensure codec works
                            test_img = cv2.resize(first_img, (width, height))
                            temp_writer.write(test_img)
                            temp_writer.release()
                            
                            # If we got here, the codec works, create the real writer
                            video_writer = cv2.VideoWriter(str(mp4_path), fourcc, MP4_FPS, (width, height))
                            if video_writer.isOpened():
                                used_codec = codec
                                self.detected_codec = codec  # Remember for next videos
                                if codec != MP4_CODECS[0]:
                                    print(f"      📹 Detected working codec: {codec}")
                                break
                            else:
                                video_writer = None
                        else:
                            temp_writer.release()
                            
                    except Exception:
                        # Silently try next codec
                        continue
            
            if video_writer is None:
                print(f"      ❌ No working codec found for {dicom_folder.name}")
                return False
            
            # Write all frames to video
            frames_written = 0
            for jpg_file in jpg_files:
                img = cv2.imread(str(jpg_file))
                if img is not None:
                    # Resize to ensure consistent dimensions
                    img_resized = cv2.resize(img, (width, height))
                    
                    # Ensure the image is in BGR format (OpenCV default)
                    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
                        video_writer.write(img_resized)
                        frames_written += 1
                    else:
                        # Convert grayscale to BGR if needed
                        if len(img_resized.shape) == 2:
                            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
                            video_writer.write(img_bgr)
                            frames_written += 1
                        else:
                            print(f"      ⚠️  Unexpected image format: {img_resized.shape}")
                else:
                    print(f"      ⚠️  Could not read image: {jpg_file}")
            
            # Release video writer
            video_writer.release()
            
            # Verify MP4 was created successfully and has reasonable size
            if mp4_path.exists() and mp4_path.stat().st_size > 1000:  # At least 1KB
                file_size_mb = mp4_path.stat().st_size / (1024 * 1024)
                print(f"      ✅ MP4 created: {mp4_path.name} ({frames_written} frames, {file_size_mb:.1f} MB)")
                
                # Delete JPG files if requested
                if DELETE_JPGS_AFTER_MP4:
                    jpgs_deleted = 0
                    for jpg_file in jpg_files:
                        try:
                            jpg_file.unlink()
                            jpgs_deleted += 1
                        except Exception as e:
                            print(f"      ⚠️  Could not delete {jpg_file.name}: {e}")
                    
                    if jpgs_deleted > 0:
                        print(f"      🗑️  Deleted {jpgs_deleted} JPG files")
                
                self.stats['mp4_videos_created'] += 1
                self.stats['jpgs_converted_to_mp4'] += frames_written
                return True
            else:
                print(f"      ❌ MP4 creation failed for {dicom_folder.name} (file too small or missing)")
                # Clean up failed file
                if mp4_path.exists():
                    mp4_path.unlink()
                return False
                
        except Exception as e:
            print(f"      ❌ Error creating MP4 for {dicom_folder.name}: {e}")
            return False
    
    def create_mp4_videos(self):
        """Convert multi-frame JPG folders to MP4 videos"""
        if not CREATE_MP4_FOR_MULTIFRAME or not CV2_AVAILABLE:
            return
        
        print(f"\n🎬 Creating MP4 videos from multi-frame folders...")
        
        # Process each DICOM folder
        dicom_folders = [d for d in self.images_dir.iterdir() if d.is_dir() and d.name.startswith('dicom_')]
        
        for dicom_folder in sorted(dicom_folders):
            # Find JPG files in this folder
            jpg_files = list(dicom_folder.glob('frame_*.jpg'))
            
            if len(jpg_files) > 1:
                print(f"   🎥 Processing {dicom_folder.name} ({len(jpg_files)} frames)")
                success = self._create_mp4_from_jpgs(dicom_folder, jpg_files)
                
                if success:
                    # Update corresponding metadata
                    dicom_index = int(dicom_folder.name.split('_')[1])
                    for metadata in self.metadata_list:
                        if metadata.get('dicom_index') == dicom_index:
                            metadata['has_mp4'] = True
                            metadata['mp4_filename'] = f"{dicom_folder.name}.mp4"
                            if DELETE_JPGS_AFTER_MP4:
                                metadata['frames_saved'] = 0  # JPGs were deleted
                                metadata['individual_jpgs'] = False
                            else:
                                metadata['individual_jpgs'] = True
                            break
                else:
                    # Mark as failed in metadata
                    dicom_index = int(dicom_folder.name.split('_')[1])
                    for metadata in self.metadata_list:
                        if metadata.get('dicom_index') == dicom_index:
                            metadata['has_mp4'] = False
                            metadata['mp4_filename'] = 'FAILED'
                            metadata['individual_jpgs'] = True
                            break
            else:
                # Single frame - mark in metadata
                dicom_index = int(dicom_folder.name.split('_')[1])
                for metadata in self.metadata_list:
                    if metadata.get('dicom_index') == dicom_index:
                        metadata['has_mp4'] = False
                        metadata['mp4_filename'] = 'SINGLE_FRAME'
                        metadata['individual_jpgs'] = True
                        break
        
        if self.stats['mp4_videos_created'] > 0:
            print(f"   ✅ Created {self.stats['mp4_videos_created']} MP4 videos from {self.stats['jpgs_converted_to_mp4']} frames")
        else:
            print(f"   ℹ️  No multi-frame folders found or no MP4s created")
    
    def save_info_file(self):
        """Save processing information and statistics"""
        info_path = self.output_dir / f"{self.study_name}_info.json"
        
        # Convert set to list for JSON serialization
        stats_copy = self.stats.copy()
        stats_copy['unique_metadata_keys'] = sorted(list(self.stats['unique_metadata_keys']))
        
        info = {
            'study_name': self.study_name,
            'tgz_path': str(self.tgz_path),
            'output_dir': str(self.output_dir),
            'images_dir': str(self.images_dir),
            'processing_time': datetime.now().isoformat(),
            'statistics': stats_copy,
            'configuration': {
                'temp_base_dir': TEMP_BASE_DIR,
                'output_base_dir': OUTPUT_BASE_DIR,
                'debug_keep_temp': DEBUG_KEEP_TEMP,
                'jpg_quality': JPG_QUALITY,
                'create_mp4_for_multiframe': CREATE_MP4_FOR_MULTIFRAME,
                'mp4_fps': MP4_FPS,
                'mp4_codecs': MP4_CODECS,
                'delete_jpgs_after_mp4': DELETE_JPGS_AFTER_MP4
            }
        }
        
        try:
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            
            print(f"📄 Info saved: {info_path}")
            return info_path
            
        except Exception as e:
            print(f"❌ Error saving info: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and self.temp_dir.exists():
            if not DEBUG_KEEP_TEMP:
                try:
                    shutil.rmtree(self.temp_dir)
                    print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not clean temp directory: {e}")
            else:
                print(f"🔍 Debug mode: Keeping temp directory: {self.temp_dir}")
    
    def _cleanup_failed_study(self):
        """Clean up partial output for failed studies"""
        try:
            if self.output_dir and self.output_dir.exists():
                print(f"🧹 Cleaning up failed study output: {self.output_dir}")
                shutil.rmtree(self.output_dir)
        except Exception as e:
            print(f"⚠️  Warning: Could not clean failed study output: {e}")
    
    def _write_failed_study_log(self):
        """Write study name to failed_studies.txt file"""
        try:
            failed_studies_file = Path("failed_studies.txt")
            with open(failed_studies_file, 'a') as f:
                f.write(f"{self.study_name}\n")
            print(f"📝 Added to failed_studies.txt: {self.study_name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not write to failed_studies.txt: {e}")
    
    def process(self):
        """Main processing pipeline"""
        start_time = datetime.now()
        
        print(f"🚀 Starting processing of study: {self.study_name}")
        print(f"   Source: {self.tgz_path}")
        
        try:
            # Step 1: Extract
            if not self.extract_study():
                return False
            
            # Step 2: Find DICOM files
            dicom_files = self.find_dicom_files()
            if not dicom_files:
                print("❌ No DICOM files found")
                return False
            
            # Step 3: Process each DICOM
            print(f"🔄 Processing {len(dicom_files)} DICOM files...")
            print(f"{'':>7} {'Filename':<35} | {'Shape':<20} | {'Frames':<20} | Series Description")
            print(f"{'-'*7} {'-'*35} | {'-'*20} | {'-'*20} | {'-'*20}")
            
            for i, dicom_file in enumerate(dicom_files):
                success = self.process_dicom_file(dicom_file, i)
                
                # Stop processing if corruption detected
                if self.study_failed:
                    print(f"\n🚫 STOPPING STUDY PROCESSING - Corruption detected")
                    print(f"   Processed {i+1}/{len(dicom_files)} files before failure")
                    self._cleanup_failed_study()
                    return False
            
            # Only continue if study didn't fail
            if self.study_failed:
                self._cleanup_failed_study()
                return False
            
            # Step 4: Create MP4 videos from multi-frame folders
            if CREATE_MP4_FOR_MULTIFRAME:
                self.create_mp4_videos()
            
            # Step 5: Save outputs
            print(f"\n💾 Saving outputs...")
            csv_path = self.save_csv_file()
            info_path = self.save_info_file()
            
            # Step 6: Summary
            duration = datetime.now() - start_time
            
            if self.study_failed:
                print(f"\n❌ Study processing FAILED due to corruption in {duration.total_seconds():.1f} seconds")
                print(f"   📊 Statistics:")
                print(f"      Total DICOMs found: {self.stats['total_dicoms']}")
                print(f"      Processed before failure: {self.stats['processed_dicoms']}")
                print(f"      Failed DICOMs: {self.stats['failed_dicoms']}")
                print(f"      Corruption errors: {len([e for e in self.stats['errors'] if 'CORRUPTION:' in e])}")
                
                if self.stats['errors']:
                    print(f"   🚫 Corruption details:")
                    for error in [e for e in self.stats['errors'] if 'CORRUPTION:' in e][:3]:
                        print(f"      • {error}")
                
                print(f"\n🗑️  Cleaned up partial output directory: {self.output_dir}")
                return False
            
            print(f"\n✅ Processing completed in {duration.total_seconds():.1f} seconds")
            print(f"   📊 Statistics:")
            print(f"      Total DICOMs: {self.stats['total_dicoms']}")
            print(f"      Successfully processed: {self.stats['processed_dicoms']}")
            print(f"      Failed: {self.stats['failed_dicoms']}")
            print(f"      Total JPG frames saved: {self.stats['total_frames_saved']}")
            if CREATE_MP4_FOR_MULTIFRAME:
                print(f"      MP4 videos created: {self.stats['mp4_videos_created']}")
                print(f"      JPG frames converted to MP4: {self.stats['jpgs_converted_to_mp4']}")
            print(f"      Unique metadata keys: {len(self.stats['unique_metadata_keys'])}")
            
            if self.stats['failed_dicoms'] > 0:
                success_rate = (self.stats['processed_dicoms'] / self.stats['total_dicoms'] * 100)
                print(f"      Success rate: {success_rate:.1f}%")
            
            if self.stats['errors']:
                print(f"   ⚠️  Errors encountered:")
                for error in self.stats['errors'][:5]:  # Show first 5 errors
                    print(f"      • {error}")
                if len(self.stats['errors']) > 5:
                    print(f"      ... and {len(self.stats['errors']) - 5} more errors")
            
            print(f"\n📁 Output files:")
            if csv_path:
                print(f"   📊 {csv_path}")
            if info_path:
                print(f"   📄 {info_path}")
            print(f"   🖼️  {self.images_dir} (contains {self.stats['total_frames_saved']} JPG images)")
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return False
        finally:
            # Always cleanup
            self.cleanup()

def main():
    if len(sys.argv) != 2:
        print("Usage: python dicom_to_csv_jpg.py <path_to_study.tgz>")
        print("\nConfiguration:")
        print(f"   Temp directory: {TEMP_BASE_DIR}")
        print(f"   Output directory: {OUTPUT_BASE_DIR}")
        print(f"   Auto-cleanup temp files: {not DEBUG_KEEP_TEMP}")
        print(f"   JPG quality: {JPG_QUALITY}")
        if CREATE_MP4_FOR_MULTIFRAME:
            print(f"   Create MP4 for multi-frame: {CREATE_MP4_FOR_MULTIFRAME}")
            print(f"   MP4 FPS: {MP4_FPS}")
            print(f"   MP4 codecs (in order): {', '.join(MP4_CODECS)}")
            print(f"   Delete JPGs after MP4: {DELETE_JPGS_AFTER_MP4}")
        print("\nOutput files per study:")
        print("   • study_metadata.csv - All DICOM metadata (sequences serialized as JSON)")
        if CREATE_MP4_FOR_MULTIFRAME:
            print("   • images/dicom_XXXX/dicom_XXXX.mp4 - Multi-frame videos")
            print("   • images/dicom_XXXX/frame_000.jpg - Single frames (or remaining JPGs)")
        else:
            print("   • images/dicom_XXXX/frame_YYY.jpg - Individual frames as JPG")
        print("   • study_info.json - Processing statistics")
        print("\nDirectory structure:")
        print("   study_name/")
        print("   ├── study_metadata.csv")
        print("   ├── study_info.json")
        print("   └── images/")
        print("       ├── dicom_0000/")
        if CREATE_MP4_FOR_MULTIFRAME:
            print("       │   ├── dicom_0000.mp4 (if multi-frame)")
            print("       │   └── frame_000.jpg (if single frame)")
        else:
            print("       │   ├── frame_000.jpg")
            print("       │   └── frame_001.jpg (if multi-frame)")
        print("       └── dicom_0001/")
        if CREATE_MP4_FOR_MULTIFRAME:
            print("           └── frame_000.jpg (single frame)")
        else:
            print("           └── frame_000.jpg")
        sys.exit(1)
    
    tgz_path = sys.argv[1]
    
    try:
        processor = DicomStudyProcessor(tgz_path)
        success = processor.process()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

# Example usage with GNU parallel:
# find /path/to/studies -name "*.tgz" | parallel -j 4 python dicom_to_csv_jpg.py {}
#
# Loading the data back:
# import pandas as pd
# from PIL import Image
# import numpy as np
# import cv2
# 
# # Load metadata
# df = pd.read_csv('study_metadata.csv')
# 
# # Parse a JSON sequence column (example)
# # import json; df['PerFrameFunctionalGroupsSequence'] = df['PerFrameFunctionalGroupsSequence'].apply(json.loads)
# 
# # Load a specific JPG image (RGB or grayscale)
# img = Image.open('images/dicom_0000/frame_000.jpg')
# img_array = np.array(img)  # (H, W, 3) for RGB or (H, W) for grayscale
#
# # Load MP4 video (if created)
# cap = cv2.VideoCapture('images/dicom_0001/dicom_0001.mp4')
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Process frame...
# cap.release()

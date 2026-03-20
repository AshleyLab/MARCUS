import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
from collections import defaultdict
import os
import sys

# Configuration
GRID_SIZE = (4, 4)
NUM_FRAMES = 16
FPS = 30
TARGET_TILE_SIZE = (240, 240)
NUM_TILES = 16

def load_and_filter_csv(csv_path):
    """Load CSV and exclude non-diagnostic sequences."""
    df = pd.read_csv(csv_path)
    csv_dir = Path(csv_path).parent
    
    # Exclude localizers/scouts
    localizer_patterns = [
        "LOC", "SSFSE LOC", "3-pl Loc Fiesta", "3PL SSFSE LOC", 
        "LOC FIESTA BH", "topogram", "calibration", "field map", "scout"
    ]
    mask = pd.Series([True] * len(df))
    for pattern in localizer_patterns:
        mask &= ~df['SeriesDescription'].str.contains(pattern, case=False, na=False)
    
    # Exclude 4D/volumetric time-resolved
    four_d_patterns = ["4D FLOW", "4D cine", "4D CINE", "time-resolved", "4D flow", "volumetric cine"]
    for pattern in four_d_patterns:
        mask &= ~df['SeriesDescription'].str.contains(pattern, case=False, na=False)
    
    df = df[mask].copy()
    df['image_path'] = df['image_folder'].apply(lambda x: str(csv_dir / x / 'frame_000.jpg'))
    
    return df, csv_dir

def get_slice_location(row):
    """Extract slice location from SliceLocation or ImagePositionPatient."""
    if pd.notna(row.get('SliceLocation')):
        return float(row['SliceLocation'])
    
    ipp = row.get('ImagePositionPatient', '')
    if isinstance(ipp, str) and ipp:
        try:
            coords = [float(x) for x in ipp.strip('[]()').split(',')]
            if len(coords) >= 3:
                return coords[2]
        except:
            pass
    return None

def compute_slice_id(slice_loc, spacing):
    """Round slice location to a bin for grouping (~0.5 × spacing)."""
    if slice_loc is None:
        return None
    bin_size = 0.5 * spacing if spacing > 0 else 1.0
    return round(slice_loc / bin_size) * bin_size

def classify_series(df):
    """Classify each row into series types."""
    results = {
        'is_cine': [],
        'is_sax': [],
        'is_long_axis': [],
        'long_axis_type': [],
        'is_rv': [],
        'is_de': [],
        'is_t2': [],
        'is_cine_ir': [],
        'is_phase_contrast': [],
        'pc_location': []
    }
    
    for _, row in df.iterrows():
        desc = str(row.get('SeriesDescription', '')).upper()
        
        is_cine = 'FIESTA' in desc or ('CINE' in desc and 'CINE IR' not in desc)
        results['is_cine'].append(is_cine)
        
        is_sax = 'SAX' in desc
        results['is_sax'].append(is_sax)
        
        is_2ch = '2CH' in desc or '2-CH' in desc
        is_3ch = '3CH' in desc or '3-CH' in desc
        is_4ch = '4CH' in desc or '4-CH' in desc
        is_long_axis = (is_2ch or is_3ch or is_4ch) and ('FIESTA' in desc or 'CINE' in desc)
        results['is_long_axis'].append(is_long_axis)
        
        if is_3ch:
            la_type = '3CH'
        elif is_4ch:
            la_type = '4CH'
        elif is_2ch:
            la_type = '2CH'
        else:
            la_type = None
        results['long_axis_type'].append(la_type)
        
        is_rv = ('RV' in desc or 'RVOT' in desc or 'RIGHT VENTRICLE' in desc) and ('FIESTA' in desc or 'CINE' in desc)
        results['is_rv'].append(is_rv)
        
        is_de = ('IRP-FGRE' in desc or 'LGE' in desc or 'DE' in desc or 'PSIR' in desc) and 'CINE IR' not in desc
        results['is_de'].append(is_de)
        
        is_t2 = ('T2' in desc and ('DIRFSE' in desc or 'STIR' in desc)) or ('T2' in desc and 'MAP' in desc)
        results['is_t2'].append(is_t2)
        
        is_cine_ir = 'CINE IR' in desc
        results['is_cine_ir'].append(is_cine_ir)
        
        is_pc = ('PC' in desc or 'PHASE' in desc) and 'CINE' not in desc and '4D' not in desc
        results['is_phase_contrast'].append(is_pc)
        
        pc_loc = None
        if is_pc:
            if 'PA' in desc or 'PULMONARY' in desc:
                if 'MPA' in desc or 'MAIN PA' in desc:
                    pc_loc = 'MPA'
                elif 'RPA' in desc:
                    pc_loc = 'RPA'
                elif 'LPA' in desc:
                    pc_loc = 'LPA'
                else:
                    pc_loc = 'PA'
            elif 'AO' in desc or 'AORTA' in desc or 'AORTIC' in desc:
                if 'ASC' in desc or 'ASCENDING' in desc:
                    pc_loc = 'AAo'
                elif 'DESC' in desc or 'DESCENDING' in desc:
                    pc_loc = 'DAo'
                else:
                    pc_loc = 'Ao'
        results['pc_location'].append(pc_loc)
    
    for key, values in results.items():
        df[key] = values
    
    return df

def add_slice_info(df):
    """Add slice location and ID to dataframe."""
    slice_locs = []
    slice_ids = []
    
    for _, row in df.iterrows():
        slice_loc = get_slice_location(row)
        slice_locs.append(slice_loc)
        
        spacing = row.get('SpacingBetweenSlices', 1.0)
        if pd.isna(spacing):
            spacing = 1.0
        slice_id = compute_slice_id(slice_loc, spacing)
        slice_ids.append(slice_id)
    
    df['slice_location'] = slice_locs
    df['slice_id'] = slice_ids
    
    return df

def sort_temporal(group_df):
    """Sort frames temporally by TriggerTime, then InstanceNumber."""
    df = group_df.copy()
    
    sort_keys = []
    if 'TriggerTime' in df.columns:
        df['TriggerTime_sort'] = pd.to_numeric(df['TriggerTime'], errors='coerce').fillna(999999)
        sort_keys.append('TriggerTime_sort')
    
    if 'InstanceNumber' in df.columns:
        df['InstanceNumber_sort'] = pd.to_numeric(df['InstanceNumber'], errors='coerce').fillna(999999)
        sort_keys.append('InstanceNumber_sort')
    
    if sort_keys:
        df = df.sort_values(sort_keys)
    
    return df

def deduplicate_series(df):
    """Deduplicate near-identical runs, keeping the best quality."""
    if len(df) == 0:
        return pd.DataFrame()
    
    best_series = {}
    
    for series_uid in df['SeriesInstanceUID'].unique():
        series_data = df[df['SeriesInstanceUID'] == series_uid]
        num_phases = len(series_data)
        desc_key = series_data.iloc[0].get('SeriesDescription', '')
        
        if desc_key not in best_series or num_phases > best_series[desc_key]['score']:
            best_series[desc_key] = {
                'score': num_phases,
                'data': series_data
            }
    
    if len(best_series) > 0:
        result = pd.concat([v['data'] for v in best_series.values()], ignore_index=True)
        return result
    return pd.DataFrame()

def get_unique_slices_for_series(df_series):
    """Get all unique slice IDs for a cine series, returning dict of slice_id -> data."""
    slices = {}
    
    for slice_id in df_series['slice_id'].dropna().unique():
        slice_data = df_series[df_series['slice_id'] == slice_id]
        
        # Get best series for this slice
        best_frames = None
        max_phases = 0
        
        for series_uid in slice_data['SeriesInstanceUID'].unique():
            series_frames = slice_data[slice_data['SeriesInstanceUID'] == series_uid]
            sorted_frames = sort_temporal(series_frames)
            
            if len(sorted_frames) > max_phases:
                max_phases = len(sorted_frames)
                best_frames = sorted_frames
        
        if best_frames is not None:
            slices[slice_id] = best_frames
    
    return slices

# Priority 1: Tissue characterization
def get_tissue_characterization(df):
    """Get DE and T2 images (up to 4 static tiles)."""
    result = []
    
    # SAX DE: basal/mid/apical
    sax_de = df[(df['is_de']) & (df['is_sax']) & (df['slice_id'].notna())].copy()
    
    if len(sax_de) > 0:
        unique_slices = sorted(sax_de['slice_id'].unique(), reverse=True)
        
        if len(unique_slices) >= 3:
            selected_slices = [unique_slices[0], unique_slices[len(unique_slices)//2], unique_slices[-1]]
            positions = ['Basal', 'Mid', 'Apical']
        elif len(unique_slices) == 2:
            selected_slices = unique_slices
            positions = ['Basal', 'Apical']
        else:
            selected_slices = unique_slices
            positions = ['Mid']
        
        for slice_id, pos in zip(selected_slices, positions):
            slice_frames = sax_de[sax_de['slice_id'] == slice_id]
            result.append({
                'type': 'DE',
                'frames': [slice_frames.iloc[0]['image_path']],
                'label': f'DE SAX\n{pos}',
                'is_static': True,
                'priority': 1
            })
    
    # Fallback: LA DE if no SAX DE
    if len(result) == 0:
        la_de = df[(df['is_de']) & (df['is_long_axis'])].copy()
        if len(la_de) > 0:
            for la_type in ['3CH', '4CH', '2CH']:
                if len(result) >= 2:
                    break
                la_data = la_de[la_de['long_axis_type'] == la_type]
                if len(la_data) > 0:
                    result.append({
                        'type': 'DE',
                        'frames': [la_data.iloc[0]['image_path']],
                        'label': f'DE {la_type}',
                        'is_static': True,
                        'priority': 1
                    })
    
    # Edema: T2-w/T2-map (mid)
    if len(result) < 4:
        t2_data = df[(df['is_t2']) & (df['is_sax']) & (df['slice_id'].notna())].copy()
        
        if len(t2_data) == 0:
            t2_data = df[(df['is_t2'])].copy()
        
        if len(t2_data) > 0:
            if 'slice_id' in t2_data.columns and t2_data['slice_id'].notna().any():
                unique_slices = sorted(t2_data[t2_data['slice_id'].notna()]['slice_id'].unique())
                mid_slice_id = unique_slices[len(unique_slices)//2]
                t2_frame = t2_data[t2_data['slice_id'] == mid_slice_id].iloc[0]
            else:
                t2_frame = t2_data.iloc[0]
            
            result.append({
                'type': 'T2',
                'frames': [t2_frame['image_path']],
                'label': 'T2 Mid',
                'is_static': True,
                'priority': 1
            })
    
    return result

# Priority 2: Long-axis & RV function (slice-aware)
def get_la_rv_core_and_extra(df):
    """Get core LV/RV clips (one best slice per view) + extra offsets."""
    core = []
    extra = []
    
    # LV long-axis: 2CH, 3CH, 4CH
    la_cine = df[(df['is_long_axis']) & (df['is_cine'])].copy()
    
    for la_type in ['2CH', '3CH', '4CH']:
        la_data = la_cine[la_cine['long_axis_type'] == la_type]
        
        if len(la_data) == 0:
            continue
        
        # Get all unique slices for this view
        slices = get_unique_slices_for_series(la_data)
        
        if len(slices) == 0:
            continue
        
        # Pick best slice (most phases)
        best_slice_id = max(slices.items(), key=lambda x: len(x[1]))[0]
        best_frames = slices[best_slice_id]
        
        indices = np.linspace(0, len(best_frames) - 1, NUM_FRAMES).astype(int)
        selected_frames = best_frames.iloc[indices]
        
        core.append({
            'type': f'{la_type} CINE',
            'frames': selected_frames['image_path'].tolist(),
            'label': f'{la_type}\nCINE',
            'is_static': False,
            'priority': 2,
            'slice_id': best_slice_id
        })
        
        # Extra slices
        for slice_id, frames in slices.items():
            if slice_id != best_slice_id:
                indices = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
                selected_frames = frames.iloc[indices]
                
                extra.append({
                    'type': f'{la_type} Extra',
                    'frames': selected_frames['image_path'].tolist(),
                    'label': f'{la_type}\nOffset',
                    'is_static': False,
                    'priority': 4,
                    'is_extra_offset': True,
                    'slice_id': slice_id
                })
    
    # RV cine
    rv_cine = df[(df['is_rv']) & (df['is_cine'])].copy()
    
    if len(rv_cine) > 0:
        # Prioritize RVOT
        rvot = rv_cine[rv_cine['SeriesDescription'].str.contains('RVOT', case=False, na=False)]
        if len(rvot) > 0:
            rv_data = rvot
            label_prefix = 'RVOT'
        else:
            rv_data = rv_cine
            label_prefix = 'RV'
        
        # Get all unique slices
        slices = get_unique_slices_for_series(rv_data)
        
        if len(slices) > 0:
            # Pick best slice
            best_slice_id = max(slices.items(), key=lambda x: len(x[1]))[0]
            best_frames = slices[best_slice_id]
            
            indices = np.linspace(0, len(best_frames) - 1, NUM_FRAMES).astype(int)
            selected_frames = best_frames.iloc[indices]
            
            core.append({
                'type': 'RV CINE',
                'frames': selected_frames['image_path'].tolist(),
                'label': f'{label_prefix}\nCINE',
                'is_static': False,
                'priority': 2,
                'slice_id': best_slice_id
            })
            
            # Extra RV slices
            for slice_id, frames in slices.items():
                if slice_id != best_slice_id:
                    indices = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
                    selected_frames = frames.iloc[indices]
                    
                    extra.append({
                        'type': 'RV Extra',
                        'frames': selected_frames['image_path'].tolist(),
                        'label': f'{label_prefix}\nOffset',
                        'is_static': False,
                        'priority': 4,
                        'is_extra_offset': True,
                        'slice_id': slice_id
                    })
    
    return core, extra

# Priority 3: Hemodynamics (2D phase-contrast)
def get_phase_contrast(df):
    """Get Ao and MPA PC (2 clips) + optional branch for coverage."""
    core = []
    branch = []
    
    pc_data = df[df['is_phase_contrast']].copy()
    
    if len(pc_data) == 0:
        return core, branch
    
    # Core: Ao and MPA
    for loc in ['AAo', 'Ao', 'MPA', 'PA']:
        if len(core) >= 2:
            break
        
        loc_data = pc_data[pc_data['pc_location'] == loc]
        if len(loc_data) == 0:
            continue
        
        # Get best series
        best_series = None
        max_phases = 0
        for series_uid in loc_data['SeriesInstanceUID'].unique():
            series_data = loc_data[loc_data['SeriesInstanceUID'] == series_uid]
            sorted_frames = sort_temporal(series_data)
            if len(sorted_frames) > max_phases:
                max_phases = len(sorted_frames)
                best_series = sorted_frames
        
        if best_series is not None:
            if len(best_series) > 1:
                indices = np.linspace(0, len(best_series) - 1, NUM_FRAMES).astype(int)
                selected_frames = best_series.iloc[indices]
                frames = selected_frames['image_path'].tolist()
                is_static = False
            else:
                frames = [best_series.iloc[0]['image_path']]
                is_static = True
            
            core.append({
                'type': 'PC',
                'frames': frames,
                'label': f'PC {loc}',
                'is_static': is_static,
                'priority': 3,
                'pc_location': loc
            })
    
    # Branch PC (RPA or LPA) for coverage pool
    for loc in ['RPA', 'LPA']:
        if len(branch) >= 1:
            break
        
        loc_data = pc_data[pc_data['pc_location'] == loc]
        if len(loc_data) == 0:
            continue
        
        best_series = None
        max_phases = 0
        for series_uid in loc_data['SeriesInstanceUID'].unique():
            series_data = loc_data[loc_data['SeriesInstanceUID'] == series_uid]
            sorted_frames = sort_temporal(series_data)
            if len(sorted_frames) > max_phases:
                max_phases = len(sorted_frames)
                best_series = sorted_frames
        
        if best_series is not None:
            if len(best_series) > 1:
                indices = np.linspace(0, len(best_series) - 1, NUM_FRAMES).astype(int)
                selected_frames = best_series.iloc[indices]
                frames = selected_frames['image_path'].tolist()
                is_static = False
            else:
                frames = [best_series.iloc[0]['image_path']]
                is_static = True
            
            branch.append({
                'type': 'PC Branch',
                'frames': frames,
                'label': f'PC {loc}',
                'is_static': is_static,
                'priority': 4,
                'is_branch_pc': True,
                'pc_location': loc
            })
    
    return core, branch

# Priority 4: Short-axis core + coverage
def get_sax_core_and_coverage(df):
    """Get SAX mid (core) + all other unique slices for coverage."""
    core = []
    coverage_pool = []
    
    sax_cine = df[(df['is_cine']) & (df['is_sax']) & (df['slice_id'].notna())].copy()
    
    if len(sax_cine) == 0:
        return core, coverage_pool
    
    sax_cine = deduplicate_series(sax_cine)
    
    # Get all unique slices across all SAX series
    all_slices = get_unique_slices_for_series(sax_cine)
    
    if len(all_slices) == 0:
        return core, coverage_pool
    
    # Sort slices base to apex
    sorted_slice_ids = sorted(all_slices.keys(), reverse=True)
    
    # Core: mid-ventricle
    mid_idx = len(sorted_slice_ids) // 2
    mid_slice_id = sorted_slice_ids[mid_idx]
    mid_frames = all_slices[mid_slice_id]
    
    indices = np.linspace(0, len(mid_frames) - 1, NUM_FRAMES).astype(int)
    selected_frames = mid_frames.iloc[indices]
    
    core.append({
        'type': 'SAX CINE',
        'frames': selected_frames['image_path'].tolist(),
        'label': 'SAX Mid',
        'is_static': False,
        'priority': 4,
        'level': 'mid',
        'slice_id': mid_slice_id
    })
    
    # Coverage: all other unique slices
    for i, slice_id in enumerate(sorted_slice_ids):
        if slice_id == mid_slice_id:
            continue
        
        frames = all_slices[slice_id]
        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
        selected_frames = frames.iloc[indices]
        
        # Determine level
        if i < len(sorted_slice_ids) // 3:
            level = 'basal'
            label = 'SAX Basal'
        elif i > 2 * len(sorted_slice_ids) // 3:
            level = 'apical'
            label = 'SAX Apical'
        else:
            level = 'mid'
            label = f'SAX Mid-{i}'
        
        coverage_pool.append({
            'type': 'SAX CINE',
            'frames': selected_frames['image_path'].tolist(),
            'label': label,
            'is_static': False,
            'priority': 4,
            'level': level,
            'is_sax_coverage': True,
            'slice_id': slice_id
        })
    
    return core, coverage_pool

# Priority 5: Optional flex
def get_optional_flex(df):
    """Get CINE-IR, perfusion, axial."""
    result = []
    
    # CINE-IR (max 1)
    cine_ir = df[(df['is_cine_ir']) & (df['is_sax']) & (df['slice_id'].notna())].copy()
    if len(cine_ir) > 0:
        unique_slices = sorted(cine_ir['slice_id'].unique())
        mid_slice_id = unique_slices[len(unique_slices)//2]
        slice_frames = cine_ir[cine_ir['slice_id'] == mid_slice_id]
        sorted_frames = sort_temporal(slice_frames)
        
        indices = np.linspace(0, len(sorted_frames) - 1, NUM_FRAMES).astype(int)
        selected_frames = sorted_frames.iloc[indices]
        
        result.append({
            'type': 'CINE-IR',
            'frames': selected_frames['image_path'].tolist(),
            'label': 'CINE-IR\nMid',
            'is_static': False,
            'priority': 5,
            'flex_type': 'cine_ir'
        })
    
    # Perfusion
    perf_data = df[df['SeriesDescription'].str.contains('perf', case=False, na=False)].copy()
    
    if len(perf_data) > 0:
        stress = perf_data[perf_data['SeriesDescription'].str.contains('stress', case=False, na=False)]
        if len(stress) > 0:
            result.append({
                'type': 'Perfusion',
                'frames': [stress.iloc[0]['image_path']],
                'label': 'Perf\nStress',
                'is_static': True,
                'priority': 5,
                'flex_type': 'perfusion'
            })
        
        rest = perf_data[perf_data['SeriesDescription'].str.contains('rest', case=False, na=False)]
        if len(rest) > 0 and len([r for r in result if r.get('flex_type') == 'perfusion']) < 2:
            result.append({
                'type': 'Perfusion',
                'frames': [rest.iloc[0]['image_path']],
                'label': 'Perf\nRest',
                'is_static': True,
                'priority': 5,
                'flex_type': 'perfusion'
            })
    
    # Axial overview
    ax_data = df[df['SeriesDescription'].str.contains('AX', case=False, na=False) & 
                df['SeriesDescription'].str.contains('FIESTA', case=False, na=False)].copy()
    
    if len(ax_data) > 0:
        series_uid = ax_data['SeriesInstanceUID'].iloc[0]
        series_data = ax_data[ax_data['SeriesInstanceUID'] == series_uid]
        sorted_frames = sort_temporal(series_data)
        
        result.append({
            'type': 'Axial',
            'frames': [sorted_frames.iloc[0]['image_path']],
            'label': 'Axial\nOverview',
            'is_static': True,
            'priority': 5,
            'flex_type': 'axial'
        })
    
    return result

def smart_allocate_coverage(sax_pool, extra_offsets, branch_pc, available_slots):
    """
    Allocate coverage with caps and 2:1 balance.
    
    Caps:
    - Extra LA/RV offsets: max 2 total (or 1 if SAX would be <4)
    - Branch PC: max 1
    - SAX coverage: target ≥4 distinct levels
    """
    result = []
    
    # Ensure diversity: ≥1 basal and ≥1 apical SAX
    has_basal = False
    has_apical = False
    
    for tile in sax_pool:
        if tile.get('level') == 'basal' and not has_basal:
            result.append(tile)
            has_basal = True
        elif tile.get('level') == 'apical' and not has_apical:
            result.append(tile)
            has_apical = True
    
    remaining_slots = available_slots - len(result)
    sax_remaining = [t for t in sax_pool if t not in result]
    
    # Determine caps for extra offsets
    # If we can get ≥4 SAX and still have room for 2 extra offsets, allow 2
    # Otherwise cap at 1
    potential_sax_count = len(result) + len(sax_remaining)
    if potential_sax_count >= 4 and remaining_slots >= 2:
        max_extra_offsets = min(2, len(extra_offsets))
    else:
        max_extra_offsets = min(1, len(extra_offsets))
    
    # Branch PC: max 1
    max_branch_pc = min(1, len(branch_pc))
    
    # Calculate targets for 2:1 SAX : (extra offsets + branch PC) ratio
    total_non_sax = max_extra_offsets + max_branch_pc
    
    # Aim for 2:1 ratio
    target_sax = min(len(sax_remaining), max(remaining_slots - total_non_sax, int(remaining_slots * 0.67)))
    
    # Ensure we get ≥4 SAX total if possible
    current_sax_count = len([t for t in result if t.get('type') == 'SAX CINE'])
    if current_sax_count + target_sax < 4:
        target_sax = min(len(sax_remaining), 4 - current_sax_count)
    
    # Add SAX
    result.extend(sax_remaining[:target_sax])
    remaining_slots -= target_sax
    
    # Add extra offsets (respecting cap)
    actual_extra = min(max_extra_offsets, remaining_slots)
    result.extend(extra_offsets[:actual_extra])
    remaining_slots -= actual_extra
    
    # Add branch PC (respecting cap)
    actual_branch = min(max_branch_pc, remaining_slots)
    result.extend(branch_pc[:actual_branch])
    
    return result[:available_slots]

def drop_tiles_to_fit(all_tiles, target_count):
    """Drop tiles in priority order if over target."""
    if len(all_tiles) <= target_count:
        return all_tiles
    
    remaining = all_tiles.copy()
    
    # 1. Axial overview
    remaining = [t for t in remaining if t.get('flex_type') != 'axial']
    if len(remaining) <= target_count:
        return remaining[:target_count]
    
    # 2. Extra perfusion/PC beyond caps
    perf = [t for t in remaining if t.get('flex_type') == 'perfusion']
    if len(perf) > 2:
        remaining = [t for t in remaining if t not in perf[2:]]
    
    branch_pc = [t for t in remaining if t.get('is_branch_pc', False)]
    if len(branch_pc) > 1:
        remaining = [t for t in remaining if t not in branch_pc[1:]]
    
    if len(remaining) <= target_count:
        return remaining[:target_count]
    
    # 3. Extra LA/RV offsets beyond cap
    extra_offsets = [t for t in remaining if t.get('is_extra_offset', False)]
    if len(extra_offsets) > 2:
        remaining = [t for t in remaining if t not in extra_offsets[2:]]
    
    if len(remaining) <= target_count:
        return remaining[:target_count]
    
    # 4. Excess SAX coverage (keep balanced)
    sax_coverage = [t for t in remaining if t.get('is_sax_coverage', False)]
    if len(sax_coverage) > 4:
        remaining = [t for t in remaining if t not in sax_coverage[4:]]
    
    if len(remaining) <= target_count:
        return remaining[:target_count]
    
    # 5. CINE-IR
    remaining = [t for t in remaining if t.get('flex_type') != 'cine_ir']
    
    return remaining[:target_count]

def backfill_to_16(all_tiles, sax_pool, extra_offsets):
    """Backfill if under 16 tiles."""
    if len(all_tiles) >= NUM_TILES:
        return all_tiles
    
    result = all_tiles.copy()
    needed = NUM_TILES - len(result)
    
    # Track used slices
    used_slices = set()
    for tile in result:
        if 'slice_id' in tile:
            used_slices.add(tile['slice_id'])
    
    # Add new SAX slices
    for tile in sax_pool:
        if needed <= 0:
            break
        if tile.get('slice_id') not in used_slices:
            result.append(tile)
            used_slices.add(tile.get('slice_id'))
            needed -= 1
    
    # Add extra offsets (respecting balance)
    for tile in extra_offsets:
        if needed <= 0:
            break
        if tile.get('slice_id') not in used_slices:
            result.append(tile)
            used_slices.add(tile.get('slice_id'))
            needed -= 1
    
    # Duplicate high-yield clips
    while len(result) < NUM_TILES and len(result) > 0:
        sax_mid = [t for t in result if t['type'] == 'SAX CINE' and 'Mid' in t.get('label', '')]
        if sax_mid:
            dup = sax_mid[0].copy()
            dup['label'] = dup['label'] + ' (dup)'
            result.append(dup)
        else:
            cine_tiles = [t for t in result if not t.get('is_static', False)]
            if cine_tiles:
                dup = cine_tiles[0].copy()
                dup['label'] = dup['label'] + ' (dup)'
                result.append(dup)
            else:
                break
    
    return result

def create_placeholder(size, text):
    """Create a placeholder image with text."""
    img = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill=255, font=font)
    
    return np.array(img)

def load_and_resize_image(image_path, target_size):
    """Load and resize an image."""
    if image_path is None or not Path(image_path).exists():
        return None
    
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    return np.array(img)

def add_label_to_image(img_array, label):
    """Add text label to image."""
    if not label:
        return img_array
    
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    lines = label.split('\n')
    
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    
    max_width = max(line_widths) if line_widths else 0
    total_height = sum(line_heights) + (len(lines) - 1) * 2
    
    padding = 4
    bg_bbox = [2, 2, max_width + padding * 2 + 2, total_height + padding * 2 + 2]
    
    overlay = img.copy()
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(bg_bbox, fill=0)
    
    img = Image.blend(img, overlay, alpha=0.6)
    draw = ImageDraw.Draw(img)
    
    y_offset = padding + 2
    for i, line in enumerate(lines):
        draw.text((padding + 2, y_offset), line, fill=255, font=font)
        y_offset += line_heights[i] + 2
    
    return np.array(img)

def create_grid_frame(tiles):
    """Create a 4x4 grid from 16 tile images."""
    rows = []
    for i in range(4):
        row_tiles = tiles[i*4:(i+1)*4]
        rows.append(np.hstack(row_tiles))
    grid = np.vstack(rows)
    
    return grid

def main(csv_path, output_video):
    print(f"Processing study: {Path(csv_path).parent.name}")
    print("Loading CSV...")
    df, csv_dir = load_and_filter_csv(csv_path)

    print("All series: ")
    print(df['SeriesDescription'].value_counts())
    
    print("\nClassifying series...")
    df = classify_series(df)
    
    print("Adding slice information...")
    df = add_slice_info(df)
    
    print("\n=== Building grid with slice-aware priority system ===\n")
    
    all_tiles = []
    
    # Priority 1: Tissue characterization
    tissue_tiles = get_tissue_characterization(df)
    print(f"Priority 1 - Tissue: {len(tissue_tiles)} tiles")
    for tile in tissue_tiles:
        print(f"  - {tile['label'].replace(chr(10), ' ')}")
    all_tiles.extend(tissue_tiles)
    
    # Priority 2: Long-axis & RV function
    la_rv_core, la_rv_extra = get_la_rv_core_and_extra(df)
    print(f"\nPriority 2 - LA/RV core: {len(la_rv_core)} tiles")
    for tile in la_rv_core:
        print(f"  - {tile['label'].replace(chr(10), ' ')}")
    all_tiles.extend(la_rv_core)
    
    # Priority 3: Hemodynamics
    pc_core, pc_branch = get_phase_contrast(df)
    print(f"\nPriority 3 - PC (Ao+PA): {len(pc_core)} tiles")
    for tile in pc_core:
        print(f"  - {tile['label'].replace(chr(10), ' ')}")
    all_tiles.extend(pc_core)
    
    # Priority 4: SAX core + coverage
    sax_core, sax_pool = get_sax_core_and_coverage(df)
    print(f"\nPriority 4 - SAX core: {len(sax_core)} tiles")
    for tile in sax_core:
        print(f"  - {tile['label'].replace(chr(10), ' ')}")
    all_tiles.extend(sax_core)
    
    # Coverage pool
    print(f"\nCoverage pool:")
    print(f"  SAX: {len(sax_pool)}, LA/RV offsets: {len(la_rv_extra)}, Branch PC: {len(pc_branch)}")
    
    # Combine all coverage candidates
    all_coverage = la_rv_extra + pc_branch
    
    available_for_coverage = NUM_TILES - len(all_tiles)
    coverage_tiles = smart_allocate_coverage(sax_pool, la_rv_extra, pc_branch, available_for_coverage)
    
    print(f"  Allocated: {len(coverage_tiles)} tiles")
    for tile in coverage_tiles:
        print(f"    - {tile['label'].replace(chr(10), ' ')}")
    all_tiles.extend(coverage_tiles)
    
    # Priority 5: Optional flex
    if len(all_tiles) < NUM_TILES:
        flex_tiles = get_optional_flex(df)
        available_flex = NUM_TILES - len(all_tiles)
        flex_tiles = flex_tiles[:available_flex]
        
        print(f"\nPriority 5 - Flex: {len(flex_tiles)} tiles")
        for tile in flex_tiles:
            print(f"  - {tile['label'].replace(chr(10), ' ')}")
        all_tiles.extend(flex_tiles)
    
    # Adjust to exactly 16
    if len(all_tiles) > NUM_TILES:
        print(f"\nOver capacity ({len(all_tiles)}), dropping tiles...")
        all_tiles = drop_tiles_to_fit(all_tiles, NUM_TILES)
    elif len(all_tiles) < NUM_TILES:
        print(f"\nUnder capacity ({len(all_tiles)}), backfilling...")
        # Gather unused coverage for backfill
        unused_sax = [t for t in sax_pool if t not in all_tiles]
        unused_extra = [t for t in la_rv_extra if t not in all_tiles]
        all_tiles = backfill_to_16(all_tiles, unused_sax, unused_extra)
    
    print(f"\n=== Final grid: {len(all_tiles)} tiles ===\n")
    
    # Pad with placeholders
    while len(all_tiles) < NUM_TILES:
        all_tiles.append({
            'type': 'Empty',
            'frames': None,
            'label': '',
            'is_static': True
        })
    
    print("Loading and resizing images...")
    
    tile_frame_lists = []
    for seq in all_tiles:
        label = seq.get('label', '')
        
        if seq.get('frames') is None or len(seq.get('frames', [])) == 0:
            tile_frames = [create_placeholder(TARGET_TILE_SIZE, label if label else 'Empty')]
        elif seq.get('is_static', False) or len(seq['frames']) == 1:
            img = load_and_resize_image(seq['frames'][0], TARGET_TILE_SIZE)
            if img is None:
                img = create_placeholder(TARGET_TILE_SIZE, label if label else 'N/A')
            else:
                img = add_label_to_image(img, label)
            tile_frames = [img]
        else:
            tile_frames = []
            for frame_path in seq['frames']:
                img = load_and_resize_image(frame_path, TARGET_TILE_SIZE)
                if img is not None:
                    img = add_label_to_image(img, label)
                    tile_frames.append(img)
            
            if len(tile_frames) == 0:
                tile_frames = [create_placeholder(TARGET_TILE_SIZE, label if label else 'N/A')]
        
        tile_frame_lists.append(tile_frames)
    
    print("Creating video...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    grid_height = TARGET_TILE_SIZE[1] * GRID_SIZE[0]
    grid_width = TARGET_TILE_SIZE[0] * GRID_SIZE[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frames_per_image = 15  # Each image shows for 15 frames = 0.5 seconds at 30fps

    out = cv2.VideoWriter(output_video, fourcc, fps, (grid_width, grid_height), isColor=False)

    for i in range(NUM_FRAMES):
        tiles = []
        for tile_frames in tile_frame_lists:
            frame_idx = i % len(tile_frames)
            tiles.append(tile_frames[frame_idx])
        
        grid = create_grid_frame(tiles)
        
        # Write the same frame multiple times
        for _ in range(frames_per_image):
            out.write(grid)

    out.release()
    
    print(f"\n✓ Video saved to: {output_video}")
    print(f"  Resolution: {grid_width}x{grid_height}")
    print(f"  Frames: {NUM_FRAMES}, FPS: {fps}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cmr_grid_temp.py <study_name>")
        print("Example: python cmr_grid_temp.py EA6467866-EA6467c49")
        sys.exit(1)
    
    study = sys.argv[1]
    csv_path = f"/home/masadi/projects_link/cmr_extracted/{study}/{study}_metadata.csv"
    output_video = f"/home/masadi/test_UCSF/{study}_grid.mp4"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    main(csv_path, output_video)

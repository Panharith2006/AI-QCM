from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class GridCell:
    # Represents a cell in the detected grid.
    x1: int
    y1: int
    x2: int
    y2: int
    row: int
    col: int
    crop: np.ndarray = None


class GridDetector:
    # Detect grid lines and extract individual cells from answer boxes.
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def detect_grid(
        self,
        image: np.ndarray,
        min_line_length: int = 50,
        max_line_gap: int = 10,
    ) -> List[GridCell]:
        
        h, w = image.shape[:2]
        
        if self.debug:
            print(f"detect_grid called with image shape: {image.shape}")
        
        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Adaptive threshold to binarize
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=10
        )
        
        if self.debug:
            edge_count = np.count_nonzero(binary)
            print(f" Binary pixels: {edge_count}")
        
        # Step 3: Extract horizontal lines
        cols = binary.shape[1]
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, cols // 20), 1))
        horizontal = binary.copy()
        horizontal = cv2.erode(horizontal, h_kernel, iterations=1)
        horizontal = cv2.dilate(horizontal, h_kernel, iterations=1)
        
        # Step 4: Extract vertical lines
        rows = binary.shape[0]
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, rows // 20)))
        vertical = binary.copy()
        vertical = cv2.erode(vertical, v_kernel, iterations=1)
        vertical = cv2.dilate(vertical, v_kernel, iterations=1)
        
        if self.debug:
            h_count = np.count_nonzero(horizontal)
            v_count = np.count_nonzero(vertical)
            print(f"Horizontal line pixels: {h_count}")
            print(f"Vertical line pixels: {v_count}")
        
        # Step 5: Find intersections (grid points)
        intersections = cv2.bitwise_and(horizontal, vertical)
        
        if self.debug:
            int_count = np.count_nonzero(intersections)
            print(f"Intersection pixels: {int_count}")
        
        # If no intersections, try dilating them to make them more detectable
        if np.count_nonzero(intersections) == 0:
            if self.debug:
                print(" No intersection pixels found → Dilating and retrying")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            intersections = cv2.dilate(intersections, kernel, iterations=2)
            if self.debug:
                int_count = np.count_nonzero(intersections)
                print(f"After dilation - Intersection pixels: {int_count}")
        
        # Step 6: Get contours at intersection points
        contours, _ = cv2.findContours(
            intersections,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if self.debug:
            print(f"Found {len(contours)} contours")
        
        if not contours or len(contours) < 4:
            if self.debug:
                print(f"Not enough contours ({len(contours)}) → Using fallback uniform split")
            fallback = self._fallback_grid_split(image)
            if self.debug:
                print(f"Fallback created {len(fallback)} cells")
            return fallback
        
        # Step 7: Get center points of intersection contours
        points = []
        for c in contours:
            x, y, w_c, h_c = cv2.boundingRect(c)
            # Use center of bounding rect
            cx = x + w_c // 2
            cy = y + h_c // 2
            points.append((cx, cy))
        
        if self.debug:
            print(f"Found {len(points)} intersection points")
        
        if len(points) < 4:
            if self.debug:
                print(f"Too few points ({len(points)}) → Using fallback")
            return self._fallback_grid_split(image)
        
        # Step 8: Sort points (top-to-bottom, left-to-right)
        points = sorted(points, key=lambda p: (p[1], p[0]))
        
        # Step 9: Group points into rows - use adaptive threshold
        # Estimate row height from image
        estimated_row_height = h // 4  # Assume at least 4 rows
        row_threshold = max(10, estimated_row_height // 3)  # Adaptive threshold
        
        if self.debug:
            print(f"Image height: {h}, estimated row height: {estimated_row_height}, threshold: {row_threshold}")
        
        rows_list = []
        current_row = []
        
        for i in range(len(points)):
            if i == 0:
                current_row.append(points[i])
                continue
            
            # If Y coordinate is similar to previous point, same row
            if abs(points[i][1] - points[i-1][1]) < row_threshold:
                current_row.append(points[i])
            else:
                # New row: sort current row left-to-right and save
                if current_row:
                    rows_list.append(sorted(current_row, key=lambda p: p[0]))
                current_row = [points[i]]
        
        # Don't forget last row
        if current_row:
            rows_list.append(sorted(current_row, key=lambda p: p[0]))
        
        if self.debug:
            print(f"Grouped into {len(rows_list)} rows")
            for row_idx, row in enumerate(rows_list):
                print(f"  Row {row_idx}: {len(row)} points @ Y={row[0][1]}")
        
        if len(rows_list) < 2:
            if self.debug:
                print(f"Too few rows ({len(rows_list)}) → Using fallback")
            return self._fallback_grid_split(image)
        
        # Step 10: Extract cells from row structure
        cells = self._extract_cells_from_intersections(image, rows_list)
        
        if self.debug:
            print(f"Extracted {len(cells)} cells from grid")
        
        return cells
    
    def _find_positions(self, lines: List[Tuple], tolerance: int = 5) -> List[int]:
        if not lines:
            return []
        
        positions = [line[0] for line in lines]
        positions = sorted(set(positions))
        
        # Cluster nearby positions
        clustered = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                # Average the cluster
                clustered.append(int(np.mean(current_cluster)))
                current_cluster = [pos]
        
        if current_cluster:
            clustered.append(int(np.mean(current_cluster)))
        
        return clustered
    
    def _extract_cells_from_intersections(
        self,
        image: np.ndarray,
        rows_list: List[List[Tuple[int, int]]],
    ) -> List[GridCell]:
        cells = []
        h, w = image.shape[:2]
        
        if self.debug:
            print(f"Extracting cells from {len(rows_list)} rows")
        
        # For each pair of rows
        for row_idx in range(len(rows_list) - 1):
            current_row = rows_list[row_idx]
            next_row = rows_list[row_idx + 1]
            
            # Get min number of columns (handle slight mismatches)
            num_cols = min(len(current_row), len(next_row))
            
            if num_cols < 2:
                if self.debug:
                    print(f"Row {row_idx}: Not enough columns ({num_cols}), skipping")
                continue
            
            if self.debug:
                print(f"  Row {row_idx}→{row_idx+1}: {num_cols} columns")
            
            # For each pair of points in current row with next row
            for col_idx in range(num_cols - 1):
                x1, y1 = current_row[col_idx]
                x2, y2 = current_row[col_idx + 1]
                x3, y3 = next_row[col_idx]
                x4, y4 = next_row[col_idx + 1]
                
                # Use the bounding box of these 4 points
                cell_x1 = min(x1, x3)
                cell_y1 = min(y1, y3)
                cell_x2 = max(x2, x4)
                cell_y2 = max(y2, y4)
                
                # Skip very small cells (less than 5 pixels)
                cell_width = cell_x2 - cell_x1
                cell_height = cell_y2 - cell_y1
                
                if cell_width < 5 or cell_height < 5:
                    if self.debug:
                        print(f"    Cell ({row_idx},{col_idx}): Skipped - too small ({cell_width}x{cell_height})")
                    continue
                
                # Boundary check
                cell_x1 = max(0, cell_x1)
                cell_y1 = max(0, cell_y1)
                cell_x2 = min(w, cell_x2)
                cell_y2 = min(h, cell_y2)
                
                crop = image[cell_y1:cell_y2, cell_x1:cell_x2].copy()
                
                if crop.size == 0:
                    if self.debug:
                        print(f"    Cell ({row_idx},{col_idx}): Empty crop")
                    continue
                
                if self.debug:
                    print(f"    Cell ({row_idx},{col_idx}):  [{cell_x1}:{cell_x2}, {cell_y1}:{cell_y2}]")
                
                cells.append(GridCell(
                    x1=cell_x1, y1=cell_y1, x2=cell_x2, y2=cell_y2,
                    row=row_idx, col=col_idx,
                    crop=crop
                ))
        
        if self.debug:
            print(f"Total cells extracted: {len(cells)}")
        
        return cells
    
    def _extract_cells(
        self,
        image: np.ndarray,
        row_positions: List[int],
        col_positions: List[int],
    ) -> List[GridCell]:
        
        cells = []
        h, w = image.shape[:2]
        
        for row_idx in range(len(row_positions) - 1):
            for col_idx in range(len(col_positions) - 1):
                y1 = row_positions[row_idx]
                y2 = row_positions[row_idx + 1]
                x1 = col_positions[col_idx]
                x2 = col_positions[col_idx + 1]
                
                # Skip cells that are too small
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                
                # Boundary check
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                crop = image[y1:y2, x1:x2].copy()
                
                if crop.size == 0:
                    continue
                
                cells.append(GridCell(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    row=row_idx, col=col_idx,
                    crop=crop
                ))
        
        return cells
    
    def _fallback_grid_split(self, image: np.ndarray, num_cols: int = 4) -> List[GridCell]:
        
        h, w = image.shape[:2]
        
        if self.debug:
            print(f" Fallback uniform split: {h}x{w} → {num_cols} columns")
        
        # Validate image size
        if h < 10 or w < 10:
            if self.debug:
                print(f" Image too small for fallback split: {h}x{w}")
            return []
        
        cells = []
        step = max(10, w // num_cols)  # Ensure minimum column width of 10px
        
        for col_idx in range(num_cols):
            x1 = col_idx * step
            x2 = min(w, (col_idx + 1) * step)
            
            # For last column, use rest of image
            if col_idx == num_cols - 1:
                x2 = w
            
            # Skip if column would be too narrow
            if x2 - x1 < 5:
                if self.debug:
                    print(f"  Column {col_idx}: Skipped (too narrow: {x2-x1}px)")
                continue
            
            crop = image[0:h, x1:x2].copy()
            
            if crop.size == 0:
                if self.debug:
                    print(f"  Column {col_idx}: Empty crop")
                continue
            
            if self.debug:
                print(f"  Column {col_idx}: [{x1}:{x2}] shape={crop.shape}")
            
            cells.append(GridCell(
                x1=x1, y1=0, x2=x2, y2=h,
                row=0, col=col_idx,
                crop=crop
            ))
        
        if self.debug:
            print(f"Fallback created {len(cells)} cells")

        # If fallback still creates no cells, create a single cell of entire image
        if not cells:
            if self.debug:
                print(" Fallback created 0 cells, returning entire image as 1 cell")
            cells.append(GridCell(
                x1=0, y1=0, x2=w, y2=h,
                row=0, col=0,
                crop=image.copy()
            ))
        
        return cells
    
    def visualize_grid(self, image: np.ndarray, cells: List[GridCell]) -> np.ndarray:
        vis = image.copy()
        
        for cell in cells:
            # Draw cell boundary
            cv2.rectangle(vis, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 255, 0), 2)
            
            # Add cell label
            cv2.putText(
                vis,
                f"({cell.row},{cell.col})",
                (cell.x1 + 5, cell.y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        return vis
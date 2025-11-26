#!/usr/bin/env python3
"""Analyze RLE format from SAM3"""

# Sample RLE from the previous run
rle_string = "425275 6 426283 33 427299 46 428318 55 429337 63 430358 68 431379 72 432400 75 433423 77 434445 79"
height = 720
width = 1280
total_pixels = height * width

counts = [int(x) for x in rle_string.split()]

print(f"Image: {width}x{height} = {total_pixels} pixels")
print(f"\nRLE analysis:")
print(f"Number of values: {len(counts)}")
print(f"Values: {counts}")

# Check if these are alternating start positions and lengths
print(f"\nChecking if format is [start, length, start, length, ...]:")
for i in range(0, min(len(counts), 10), 2):
    if i+1 < len(counts):
        start = counts[i]
        length = counts[i+1]
        end = start + length
        print(f"  Segment {i//2}: start={start}, length={length}, end={end}")
        if end > total_pixels:
            print(f"    WARNING: end position {end} > total pixels {total_pixels}")

# Check if these are just run lengths (standard RLE)
print(f"\nChecking if format is standard RLE [run1, run2, run3, ...]:")
cumsum = 0
for i, count in enumerate(counts[:10]):
    cumsum += count
    print(f"  Run {i}: length={count}, cumulative={cumsum}")
    if cumsum > total_pixels:
        print(f"    WARNING: cumulative {cumsum} > total pixels {total_pixels}")
        break

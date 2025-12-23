#!/usr/bin/env python3
"""
Extract Mermaid diagrams from markdown files and render them as PNG.
"""
import re
import os
import subprocess
from pathlib import Path

def extract_mermaid_blocks(md_file):
    """Extract all Mermaid code blocks from a markdown file."""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match ```mermaid ... ```
    pattern = r'```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    return matches

def render_mermaid_to_png(mermaid_code, output_path):
    """Render Mermaid code to PNG using mmdc."""
    # Create temporary mermaid file
    temp_mmd = output_path.with_suffix('.mmd')
    with open(temp_mmd, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    
    try:
        # Render using mmdc
        subprocess.run(
            ['mmdc', '-i', str(temp_mmd), '-o', str(output_path), '-b', 'white'],
            check=True,
            capture_output=True
        )
        print(f"✓ Rendered: {output_path}")
        # Clean up temp file
        temp_mmd.unlink()
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to render {output_path}: {e.stderr.decode()}")
        temp_mmd.unlink()
        return False

def main():
    docs_dir = Path('docs')
    figures_dir = Path('thesis/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all markdown files
    md_files = list(docs_dir.glob('chapter*.md'))
    
    diagram_count = 0
    for md_file in sorted(md_files):
        print(f"\nProcessing {md_file.name}...")
        mermaid_blocks = extract_mermaid_blocks(md_file)
        
        for i, mermaid_code in enumerate(mermaid_blocks, 1):
            # Generate output filename based on chapter and diagram number
            chapter_match = re.search(r'chapter(\d+)', md_file.name)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                section_match = re.search(r'section(\d+_\d+)', md_file.name)
                if section_match:
                    section = section_match.group(1)
                    output_name = f"fig_{chapter_num}_{section}_{i}.png"
                else:
                    output_name = f"fig_{chapter_num}_{i}.png"
            else:
                output_name = f"fig_{md_file.stem}_{i}.png"
            
            output_path = figures_dir / output_name
            if render_mermaid_to_png(mermaid_code, output_path):
                diagram_count += 1
    
    print(f"\n✓ Rendered {diagram_count} diagrams to {figures_dir}")

if __name__ == '__main__':
    main()

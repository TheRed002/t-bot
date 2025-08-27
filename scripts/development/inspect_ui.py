#!/usr/bin/env python3
"""
Inspect UI layout issues using Playwright
"""

import asyncio
from playwright.async_api import async_playwright
import json
from pathlib import Path

async def inspect_ui():
    async with async_playwright() as p:
        # Launch browser in headless mode
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=1
        )
        page = await context.new_page()
        
        # Navigate to the app
        print("📱 Navigating to T-Bot frontend...")
        try:
            await page.goto('http://localhost:3000', timeout=10000)
            await page.wait_for_load_state('networkidle', timeout=10000)
        except Exception as e:
            print(f"❌ Could not connect to frontend: {e}")
            print("Make sure the frontend is running (make run-frontend)")
            await browser.close()
            return
        
        # Check if we need to login
        if await page.is_visible('text=Sign In'):
            print("🔐 Logging in...")
            await page.fill('input[name="username"]', 'admin')
            await page.fill('input[type="password"]', 'admin123')
            await page.click('button[type="submit"]')
            await page.wait_for_url('**/dashboard', timeout=5000)
        
        # Take screenshots of the issue
        print("📸 Taking screenshots...")
        screenshots_dir = Path('/mnt/e/Work/P-41 Trading/code/t-bot/ui_analysis')
        screenshots_dir.mkdir(exist_ok=True)
        
        # Capture dashboard
        await page.screenshot(path=str(screenshots_dir / 'dashboard_full.png'), full_page=True)
        
        # Analyze layout issues
        print("\n🔍 Analyzing layout issues...")
        
        # Get sidebar and main content measurements
        sidebar = await page.query_selector('[class*="MuiDrawer-root"]')
        main_content = await page.query_selector('main')
        
        layout_data = {}
        
        if sidebar:
            sidebar_box = await sidebar.bounding_box()
            layout_data['sidebar'] = sidebar_box
            print(f"Sidebar: width={sidebar_box['width']}px")
            
            # Check sidebar styles
            sidebar_styles = await page.evaluate('''() => {
                const sidebar = document.querySelector('[class*="MuiDrawer-root"]');
                if (sidebar) {
                    const styles = window.getComputedStyle(sidebar);
                    return {
                        width: styles.width,
                        position: styles.position,
                        left: styles.left,
                        transform: styles.transform
                    };
                }
                return null;
            }''')
            print(f"Sidebar styles: {sidebar_styles}")
        
        if main_content:
            main_box = await main_content.bounding_box()
            layout_data['main'] = main_box
            print(f"Main content: x={main_box['x']}px, width={main_box['width']}px")
            
            # Check main content styles
            main_styles = await page.evaluate('''() => {
                const main = document.querySelector('main');
                if (main) {
                    const styles = window.getComputedStyle(main);
                    return {
                        marginLeft: styles.marginLeft,
                        paddingLeft: styles.paddingLeft,
                        width: styles.width,
                        maxWidth: styles.maxWidth
                    };
                }
                return null;
            }''')
            print(f"Main content styles: {main_styles}")
        
        # Calculate gap
        if sidebar and main_content:
            gap = layout_data['main']['x'] - (layout_data['sidebar']['x'] + layout_data['sidebar']['width'])
            print(f"\n⚠️  Gap between sidebar and main content: {gap}px")
            
            if gap > 20:
                print(f"❌ Large gap detected! Should be around 0-20px, but is {gap}px")
        
        # Check all Box components for spacing issues
        print("\n🔍 Checking Box components spacing...")
        box_spacing = await page.evaluate('''() => {
            const boxes = document.querySelectorAll('[class*="MuiBox-root"]');
            const issues = [];
            boxes.forEach((box, index) => {
                const styles = window.getComputedStyle(box);
                const ml = parseFloat(styles.marginLeft) || 0;
                const pl = parseFloat(styles.paddingLeft) || 0;
                
                // Check for excessive margins/padding
                if (ml > 100) {
                    issues.push({
                        element: box.className,
                        issue: 'Excessive margin-left: ' + styles.marginLeft,
                        selector: box.id || box.className
                    });
                }
                if (pl > 50) {
                    issues.push({
                        element: box.className,
                        issue: 'Excessive padding-left: ' + styles.paddingLeft,
                        selector: box.id || box.className
                    });
                }
            });
            return issues;
        }''')
        
        if box_spacing:
            print("Found spacing issues:")
            for issue in box_spacing:
                print(f"  - {issue['issue']} in {issue['selector']}")
        
        # Save analysis results
        analysis_file = screenshots_dir / 'layout_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump({
                'layout_data': layout_data,
                'spacing_issues': box_spacing,
                'gap_size': gap if 'gap' in locals() else None
            }, f, indent=2)
        
        print(f"\n💾 Analysis saved to {analysis_file}")
        
        # Navigate to other pages to check consistency
        pages_to_check = [
            ('/trading', 'Trading'),
            ('/portfolio', 'Portfolio'),
            ('/bots', 'Bot Management'),
            ('/playground', 'Playground')
        ]
        
        for path, name in pages_to_check:
            print(f"\n📱 Checking {name} page...")
            await page.goto(f'http://localhost:3000{path}')
            await page.wait_for_load_state('networkidle', timeout=5000)
            
            # Take screenshot
            await page.screenshot(
                path=str(screenshots_dir / f'{name.lower().replace(" ", "_")}.png'),
                full_page=False
            )
            
            # Check main content position
            main_content = await page.query_selector('main')
            if main_content:
                main_box = await main_content.bounding_box()
                print(f"  Main content x position: {main_box['x']}px")
                if main_box['x'] > 300:
                    print(f"  ⚠️  Content too far from left edge!")
        
        await browser.close()
        print("\n✅ UI inspection complete!")

if __name__ == "__main__":
    asyncio.run(inspect_ui())
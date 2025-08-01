#!/usr/bin/env python3
# src/image_quality_fusion/tests/run_tests.py
"""
Test runner for image quality fusion tests.
Can run individual tests or all tests together.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def run_test(test_name):
    """Run a specific test by name"""
    test_functions = {
        'brisque': 'test_brisque_standalone',
        'aesthetic': 'test_original_aesthetic_standalone', 
        'clip': 'test_clip_standalone',
        'pipeline': 'test_basic_pipeline_standalone'
    }
    
    if test_name not in test_functions:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_functions.keys())}")
        return False
    
    try:
        if test_name == 'brisque':
            from src.image_quality_fusion.tests.test_brisque_opencv import test_brisque_standalone
            test_brisque_standalone()
        elif test_name == 'aesthetic':
            from src.image_quality_fusion.tests.test_aesthetic_original import test_original_aesthetic_standalone
            test_original_aesthetic_standalone()
        elif test_name == 'clip':
            from src.image_quality_fusion.tests.test_clip import test_clip_standalone
            test_clip_standalone()
        elif test_name == 'pipeline':
            from src.image_quality_fusion.tests.test_pipeline import test_basic_pipeline_standalone
            test_basic_pipeline_standalone()
        
        return True
        
    except Exception as e:
        print(f"❌ Error running {test_name} test: {e}")
        return False

def run_all_tests():
    """Run all available tests"""
    tests = ['brisque', 'aesthetic', 'clip', 'pipeline']
    results = []
    
    print("=" * 60)
    print("RUNNING ALL IMAGE QUALITY FUSION TESTS")
    print("=" * 60)
    
    for test_name in tests:
        print(f"\n{'=' * 20} {test_name.upper()} TEST {'=' * 20}")
        success = run_test(test_name)
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.capitalize():<15}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed.")
    
    return all_passed

def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        # Run all tests
        run_all_tests()
    elif len(sys.argv) == 2:
        # Run specific test
        test_name = sys.argv[1].lower()
        if test_name in ['all', 'everything']:
            run_all_tests()
        else:
            run_test(test_name)
    else:
        print("Usage:")
        print("  python run_tests.py              # Run all tests")
        print("  python run_tests.py <test_name>  # Run specific test")
        print("")
        print("Available tests: brisque, aesthetic, clip, pipeline")

if __name__ == "__main__":
    main()
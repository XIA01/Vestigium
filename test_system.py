#!/usr/bin/env python3
"""
VESTIGIUM Quick Test - Verify all 4 phases work
"""

import sys
import numpy as np
import logging

sys.path.insert(0, '/media/latin/60FD21291B249B8D8/Programacion/HP')

from src.utils import setup_logging, get_logger
from src.backend import SignalProcessor, NeuromorphicEngine, SLAMTopological

logger = get_logger("test")

def test_signal_processor():
    """Test Phase 1: Signal Processing"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Signal Processor (Phase 1)")
    logger.info("="*60)

    processor = SignalProcessor(num_routers=10, num_bands=2)

    for i in range(50):
        rssi_data = np.random.normal(-60, 5, size=(10, 2)).astype(np.float32)
        result = processor.process_rssi(rssi_data)

        if i == 45 and result["csi_virtual"] is not None:
            logger.info(f"✓ CSI Virtual shape: {result['csi_virtual'].shape}")
            logger.info(f"  Band ratio min/max: {result['band_ratio'].min():.3f} / {result['band_ratio'].max():.3f}")
            return True

    logger.error("✗ Signal Processor test failed")
    return False


def test_neuromorphic():
    """Test Phase 2: Neuromorphic Engine"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Neuromorphic Engine (Phase 2)")
    logger.info("="*60)

    engine = NeuromorphicEngine(num_neurons=256, num_particles=100)

    for i in range(30):
        csi_data = np.random.randn(153, 2).astype(np.float32)
        result = engine.process_csi_virtual(csi_data)

    logger.info(f"✓ Particles shape: {result['particle_positions'].shape}")
    logger.info(f"✓ Clusters detected: {len(result['clusters'])}")
    logger.info(f"  Mean weight: {result['particle_weights'].mean():.6f}")
    return True


def test_slam():
    """Test Phase 3: SLAM Topological"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: SLAM Topological (Phase 3)")
    logger.info("="*60)

    slam = SLAMTopological()

    for i in range(30):
        cluster = {
            "x": np.sin(i * 0.1) * 5,
            "y": np.cos(i * 0.1) * 5,
            "confidence": 0.7,
            "velocity": [0.5, 0.5],
        }
        slam.update_from_clusters([cluster])

    heatmap = slam.get_heatmap()
    logger.info(f"✓ Heatmap shape: {heatmap.shape}")
    logger.info(f"✓ Occupancy: mean={slam.get_stats()['mean_occupancy']:.3f}")
    return True


def test_all():
    """Run all tests"""
    setup_logging(level="INFO")

    logger.info("🧪 VESTIGIUM System Test Suite")

    tests = [
        ("Signal Processor", test_signal_processor),
        ("Neuromorphic Engine", test_neuromorphic),
        ("SLAM Topological", test_slam),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}", exc_info=True)
            results.append((name, False))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {name}")

    all_passed = all(p for _, p in results)
    logger.info("="*60)

    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error("⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(test_all())

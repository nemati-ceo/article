#!/usr/bin/env python3
"""
Test script to verify GPU support in paperflow
"""

print('=== GPU Availability Check ===')
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('No CUDA GPUs detected')
except ImportError:
    print('PyTorch not installed - GPU support not available')

print()
print('=== Testing paperflow GPU support ===')
try:
    from paperflow import PaperPipeline

    print('Creating pipeline with gpu=True...')
    pipeline = PaperPipeline(gpu=True)

    print('✅ Pipeline created successfully!')
    print('GPU support is enabled. If CUDA is available, Marker AI will use GPU for PDF extraction.')
    print('If no CUDA GPUs are detected above, it will gracefully fall back to CPU.')

except Exception as e:
    print(f'❌ Error creating pipeline: {e}')

print()
print('=== Testing actual PDF processing (if marker-pdf is installed) ===')
try:
    from paperflow.processors import MarkerProcessor

    print('Creating MarkerProcessor with GPU...')
    processor = MarkerProcessor(gpu=True)

    if processor.available:
        print('✅ Marker AI is available and GPU-enabled!')
        print('PDF extraction will use GPU acceleration if CUDA is available.')
    else:
        print('⚠️ Marker AI not available - install with: pip install paperflow[extraction]')

except Exception as e:
    print(f'❌ Error testing MarkerProcessor: {e}')
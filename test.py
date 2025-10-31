#!/usr/bin/env python3
"""
Troubleshooting script for CUDA errors in Task 2.
Run this to diagnose and fix common CUDA issues.
"""
import torch
import os

def check_cuda_setup():
    """Check CUDA availability and configuration."""
    print("=" * 60)
    print("CUDA Setup Diagnostics")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("   ⚠️  CUDA is not available. Make sure you have:")
        print("      - NVIDIA GPU")
        print("      - CUDA toolkit installed")
        print("      - PyTorch with CUDA support")
        return False
    
    # Check CUDA version
    print(f"2. CUDA Version: {torch.version.cuda}")
    print(f"3. PyTorch Version: {torch.__version__}")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"4. Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"\n   GPU {i}:")
        print(f"   - Name: {torch.cuda.get_device_name(i)}")
        print(f"   - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Check current memory usage
        if torch.cuda.is_available():
            print(f"   - Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"   - Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    return True


def test_basic_operations():
    """Test basic CUDA operations."""
    print("\n" + "=" * 60)
    print("Testing Basic CUDA Operations")
    print("=" * 60)
    
    try:
        # Test tensor creation
        print("\n1. Creating tensor on CUDA...")
        x = torch.randn(100, 100).cuda()
        print("   ✓ Success")
        
        # Test computation
        print("2. Testing matrix multiplication...")
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        print("   ✓ Success")
        
        # Test cleanup
        print("3. Testing memory cleanup...")
        del x, y, z
        torch.cuda.empty_cache()
        print("   ✓ Success")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_model_loading(device="cuda:0"):
    """Test loading a small model."""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("\n1. Loading tokenizer...")
        model_name = "gpt2"  # Use smaller model for testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✓ Success")
        
        print("2. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        print("   ✓ Success")
        
        print("3. Testing inference...")
        inputs = tokenizer("Hello world", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        print("   ✓ Success")
        
        print("4. Cleaning up...")
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        print("   ✓ Success")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def suggest_fixes():
    """Suggest potential fixes for common issues."""
    print("\n" + "=" * 60)
    print("Potential Fixes")
    print("=" * 60)
    
    print("\n1. Memory Issues:")
    print("   - Reduce batch size (N) from 8 to 4 or 2")
    print("   - Run: export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
    print("   - Add torch.cuda.empty_cache() more frequently")
    
    print("\n2. CUBLAS Issues:")
    print("   - Reinstall PyTorch with matching CUDA version:")
    print("     pip install torch --force-reinstall")
    print("   - Check CUDA_HOME: export CUDA_HOME=/usr/local/cuda")
    
    print("\n3. Environment Variables:")
    print("   - Set: export CUDA_LAUNCH_BLOCKING=1")
    print("   - Set: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    print("\n4. Code Modifications:")
    print("   - Use model.to('cpu') if GPU memory insufficient")
    print("   - Process particles sequentially instead of in parallel")
    print("   - Use gradient checkpointing")
    
    print("\n5. System Check:")
    print("   - Run: nvidia-smi")
    print("   - Check if other processes are using GPU")
    print("   - Restart Python kernel/session")


def main():
    """Run all diagnostics."""
    # Check CUDA setup
    cuda_ok = check_cuda_setup()
    
    if cuda_ok:
        # Test basic operations
        basic_ok = test_basic_operations()
        
        if basic_ok:
            # Test model loading
            model_ok = test_model_loading()
            
            if not model_ok:
                print("\n⚠️  Model loading failed!")
        else:
            print("\n⚠️  Basic CUDA operations failed!")
    else:
        print("\n⚠️  CUDA not available!")
    
    # Always show suggestions
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("Diagnostics Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

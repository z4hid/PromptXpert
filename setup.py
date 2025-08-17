#!/usr/bin/env python3
"""Setup script for PromptXpert development environment."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e.stderr}")
        return False


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} is not supported. Python 3.8+ required.")
        return False


def install_dependencies():
    """Install Python dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def setup_environment():
    """Setup environment file."""
    print("⚙️ Setting up environment file...")
    env_template = Path(".env.template")
    env_file = Path(".env")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if env_template.exists():
        # Copy template to .env
        with open(env_template, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✓ Created .env file from template")
        print("📝 Please edit .env file and add your GROQ_API_KEY")
        return True
    else:
        print("✗ .env.template not found")
        return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = ["outputs", "mlruns", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✓ Created necessary directories")
    return True


def verify_dataset():
    """Verify dataset exists and has correct format."""
    print("📊 Verifying dataset...")
    dataset_path = Path("data/prompts_dataset.csv")
    
    if not dataset_path.exists():
        print("✗ Dataset not found at data/prompts_dataset.csv")
        return False
    
    # Check format
    with open(dataset_path, 'r') as f:
        header = f.readline().strip()
        lines = f.readlines()
    
    required_columns = ['initial_prompt', 'optimized_prompt']
    if not all(col in header for col in required_columns):
        print(f"✗ Dataset missing required columns: {required_columns}")
        return False
    
    if len(lines) < 5:
        print("⚠️ Dataset is very small (< 5 examples). Consider adding more data.")
    
    print(f"✓ Dataset verified ({len(lines)} examples)")
    return True


def run_basic_tests():
    """Run basic structure tests."""
    print("🧪 Running basic tests...")
    return run_command("python tests/test_structure.py", "Running structure tests")


def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 PromptXpert Setup Complete!")
    print("=" * 60)
    
    print("\n📝 Next Steps:")
    print("1. Edit .env file and add your GROQ_API_KEY:")
    print("   GROQ_API_KEY=your_actual_api_key_here")
    
    print("\n2. Test the installation:")
    print("   python examples.py")
    
    print("\n3. Run training:")
    print("   python -m src.train")
    
    print("\n4. Or use the CLI:")
    print("   python -m src.cli info")
    print("   python -m src.cli train")
    
    print("\n🔧 Optional: Start MLflow UI:")
    print("   mlflow ui --port 5000")
    print("   # Then open http://localhost:5000")
    
    print("\n📚 Documentation:")
    print("   - README.md: Full documentation")
    print("   - examples.py: Usage examples")
    print("   - tests/: Test scripts")
    
    print("\n🐛 Issues?")
    print("   - Check the troubleshooting section in examples.py")
    print("   - Ensure GROQ_API_KEY is set correctly")
    print("   - Verify dataset format in data/prompts_dataset.csv")


def main():
    """Main setup function."""
    print("🚀 PromptXpert Setup Script")
    print("=" * 40)
    
    # Run setup steps
    steps = [
        check_python_version,
        install_dependencies,
        setup_environment,
        create_directories,
        verify_dataset,
        run_basic_tests,
    ]
    
    failed_steps = []
    
    for step in steps:
        if not step():
            failed_steps.append(step.__name__)
    
    if failed_steps:
        print(f"\n❌ Setup failed. Failed steps: {failed_steps}")
        print("Please fix the issues above and run setup.py again.")
        return False
    
    display_next_steps()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
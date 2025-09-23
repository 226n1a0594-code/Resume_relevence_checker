#!/usr/bin/env python3
"""
Resume Relevance Check System Setup Script
Handles Python version compatibility and package installation
"""

import sys
import subprocess
import os
import platform
from pathlib import Path

def print_banner():
    print("🚀 Resume Relevance Check System Setup")
    print("=" * 50)

def check_python_version():
    """Check Python version and recommend compatible versions"""
    version = sys.version_info
    print(f"📍 Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        print("✅ Python version is compatible!")
        return "compatible"
    
    else:
        print("❌ Python version too old. Please upgrade to Python 3.8+")
        sys.exit(1)

def create_requirements_original():
    """Create requirements.txt with working versions"""
    requirements = """streamlit>=1.28.0
streamlit-option-menu>=0.3.6
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
spacy>=3.6.0
PyMuPDF>=1.23.0
python-docx>=0.8.11
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
regex>=2023.6.0
nltk>=3.8.0
python-dateutil>=2.8.0
typing-extensions>=4.7.0
textblob>=0.17.0
wordcloud>=1.9.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("📄 Created requirements.txt with working versions")

def create_requirements_compatible():
    """Create requirements.txt with Python 3.13+ compatible versions"""
    requirements = """streamlit>=1.28.0
streamlit-option-menu>=0.3.6
pandas>=2.1.0
numpy>=1.25.0
scikit-learn>=1.3.0
tensorflow>=2.17.0
tf-keras
sentence-transformers>=2.2.0
spacy>=3.6.0
PyMuPDF>=1.23.0
python-docx>=0.8.11
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
regex>=2023.6.0
nltk>=3.8.0
python-dateutil>=2.8.0
typing-extensions>=4.7.0
textblob>=0.17.0
wordcloud>=1.9.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("📄 Created requirements.txt with Python 3.13 compatible versions")

def install_packages():
    """Install packages with error handling and TensorFlow compatibility fix"""
    print("\n📦 Installing packages...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True, text=True)
        print("✅ Updated pip")
        
        # Install TensorFlow and tf-keras for Python 3.13
        print("🔧 Installing TensorFlow and Keras compatibility packages...")
        tf_result = subprocess.run([sys.executable, "-m", "pip", "install", 
                                   "tensorflow", "tf-keras"], 
                                  capture_output=True, text=True)
        
        if tf_result.returncode == 0:
            print("✅ TensorFlow and tf-keras installed successfully!")
        else:
            print("⚠️  TensorFlow installation had issues, continuing with other packages...")
            print(f"Error: {tf_result.stderr}")
        
        # Install other packages
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully!")
            return True
        else:
            print("❌ Some packages failed to install:")
            print(result.stderr)
            print("\n🔧 Trying alternative installation method...")
            
            # Try installing packages individually
            essential_packages = [
                "streamlit", "streamlit-option-menu", "pandas", "numpy", 
                "scikit-learn", "plotly", "matplotlib", "seaborn",
                "PyMuPDF", "python-docx", "nltk", "textblob", "wordcloud"
            ]
            
            for package in essential_packages:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, capture_output=True, text=True)
                    print(f"✅ Installed {package}")
                except subprocess.CalledProcessError:
                    print(f"⚠️  Failed to install {package}")
            
            # Try sentence-transformers separately
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"], 
                             check=True, capture_output=True, text=True)
                print("✅ Installed sentence-transformers")
            except subprocess.CalledProcessError:
                print("⚠️  Failed to install sentence-transformers (will use basic similarity)")
            
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during installation: {e}")
        return False

def create_virtual_environment():
    """Create and setup virtual environment"""
    print("\n🔧 Setting up virtual environment...")
    
    # Check if venv exists
    if os.path.exists("resume_env"):
        print("📁 Virtual environment already exists")
        return True
    
    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "resume_env"], check=True)
        print("✅ Created virtual environment: resume_env")
        
        # Get activation script path
        if platform.system() == "Windows":
            activate_script = "resume_env\\Scripts\\activate.bat"
            pip_path = "resume_env\\Scripts\\pip.exe"
        else:
            activate_script = "resume_env/bin/activate"
            pip_path = "resume_env/bin/pip"
        
        print(f"📋 To activate: {activate_script}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk_downloads = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
                print(f"✅ Downloaded {item}")
            except Exception as e:
                print(f"⚠️  Could not download {item}: {e}")
        
        print("✅ NLTK data download completed!")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def download_spacy_model():
    """Download spaCy English model"""
    print("\n🌐 Downloading spaCy English model...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ spaCy English model downloaded successfully!")
            return True
        else:
            print(f"⚠️  Could not download spaCy model: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download spaCy model: {e}")
        return False

def create_project_structure():
    """Create necessary project directories and files"""
    print("\n📁 Creating project structure...")
    
    directories = [
        "uploads",
        "reports", 
        "data",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
resume_env/

# Uploads and temp files
uploads/
temp/
*.pdf
*.docx
*.doc

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✅ Created .gitignore")

def main():
    print_banner()
    
    # Check Python version
    choice = check_python_version()
    
    # Create project structure
    create_project_structure()
    
    # Create requirements with compatible versions
    create_requirements_compatible()
    
    # Option to create virtual environment
    use_venv = input("\n🤔 Create virtual environment? (recommended) [y/N]: ").lower().startswith('y')
    
    if use_venv:
        if create_virtual_environment():
            print("\n📋 Next steps:")
            if platform.system() == "Windows":
                print("   1. Run: resume_env\\Scripts\\activate")
            else:
                print("   1. Run: source resume_env/bin/activate")
            print("   2. Run: python -m pip install -r requirements.txt")
            print("   3. Run: streamlit run app.py")
            return
    
    # Install packages directly
    if install_packages():
        download_nltk_data()
        download_spacy_model()
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Open your browser to the displayed URL")
        print("   3. Start uploading resumes and job descriptions!")
        print("\n🔧 Note: Latest TensorFlow and tf-keras installed for Python 3.13 compatibility")
        
    else:
        print("\n❌ Setup failed. Try installing packages manually:")
        print("   pip install tensorflow tf-keras")
        print("   pip install streamlit streamlit-option-menu pandas numpy")
        print("   pip install sentence-transformers plotly matplotlib seaborn")
        print("   pip install PyMuPDF spacy textblob wordcloud")

if __name__ == "__main__":
    main()
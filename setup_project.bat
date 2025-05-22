@echo off
echo Setting up Takeaways model project...

:: Create main directories and subdirectories
mkdir takeaways-model
mkdir takeaways-model\config
mkdir takeaways-model\data
mkdir takeaways-model\evaluation
mkdir takeaways-model\model
mkdir takeaways-model\monitoring
mkdir takeaways-model\scripts
mkdir takeaways-model\serve
mkdir takeaways-model\.github\workflows

:: Initialize git repository
cd takeaways-model
git init

:: Create initial configuration
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Configure Hugging Face token
echo Please set your Hugging Face token as a GitHub secret named HUGGING_FACE_TOKEN

:: Setup complete message
echo.
echo Project setup completed successfully!
echo Next steps:
echo 1. Create a repository on GitHub named 'Takeaways'
echo 2. Add your Hugging Face token as a GitHub secret
echo 3. Push the code to GitHub to trigger the training workflow
echo.
pause

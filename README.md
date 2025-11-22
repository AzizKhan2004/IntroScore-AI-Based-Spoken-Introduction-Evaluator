# IntroScore-AI-Based-Spoken-Introduction-Evaluator
A web application that evaluates a student’s spoken introduction using a transcript and a rubric. The tool accepts a text transcript, loads a rubric from Excel, applies a keyword-based scoring algorithm

🚀 How to Run the Application Locally

Follow these steps to run the Spoken Introduction Evaluation System on your local machine:

1️⃣ Install Prerequisites

Ensure you have the following installed:

Python 3.9+

Git

A browser (Chrome/Edge/Firefox etc.)

Check Python version:

python --version

2️⃣ Clone the Repository
git clone https://github.com/<your-username>/spoken-intro-evaluator.git
cd spoken-intro-evaluator


Replace <your-username> with your actual GitHub username

3️⃣ Create & Activate Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate


Mac / Linux

python3 -m venv venv
source venv/bin/activate


The terminal prompt should now show (venv) meaning the environment is active.

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Prepare Rubric File

Ensure rubric.xlsx is present in the project folder.
(Example structure already shared above.)

6️⃣ Run the Flask Application
python app.py


If successful, the terminal will show:

Running on http://127.0.0.1:5000/

7️⃣ Open App in Browser

Copy & paste the URL into your browser:

http://127.0.0.1:5000/

8️⃣ Use the Web Application

On the web UI:

Paste transcript text OR upload .txt file

Upload rubric.xlsx (optional — default one will be used)

Click Score

You will see:

Overall score /100

Table with per-criterion scores

Transcript displayed

Option to evaluate another transcript

9️⃣ Stop the Server

Press CTRL + C in the terminal to stop the application.

🧩 Troubleshooting
Issue	Solution
ModuleNotFoundError: No module named flask	Run pip install -r requirements.txt
Website not opening	Make sure server is still running in terminal
Rubric not found	Ensure rubric.xlsx exists in project root or upload via UI
Optional: Reinstall Dependencies Fresh
pip uninstall -y -r requirements.txt
pip install -r requirements.txt

🎯 Done!

Your local development environment is ready and running.
Now you can modify, test, and record your demo video.

If you want, I can also provide:

✅ Screenshots to include in README
🎥 A script/template for your demo video narration
🚀 Deployment instructions for Render/Vercel/Railway
📝 A polished abstract for your case study report

Would you like me to add a “Features” section and UI screenshots for your README as well?

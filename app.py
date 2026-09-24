from flask import Flask, render_template, request, redirect, url_for, session

import sqlite3

import os

import re

import fitz

from docx import Document

from werkzeug.security import generate_password_hash, check_password_hash

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

app.secret_key = "careerai_secret_key_change_later"

DATABASE = "career_ai.db"

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------

def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn

def init_db():

    conn = get_db()

    # Users table

    conn.execute("""

        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

    """)

    # Resumes table

    conn.execute("""

        CREATE TABLE IF NOT EXISTS resumes (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            file_path TEXT NOT NULL,

            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id) REFERENCES users(id)

        )

    """)

    # Job matches table

    conn.execute("""

        CREATE TABLE IF NOT EXISTS job_matches (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            match_score INTEGER NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id) REFERENCES users(id)

        )

    """)

    # Applications table

    conn.execute("""

        CREATE TABLE IF NOT EXISTS applications (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            company TEXT NOT NULL,

            job_role TEXT NOT NULL,

            status TEXT NOT NULL DEFAULT 'Saved',

            applied_date TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id) REFERENCES users(id)

        )

    """)

    conn.commit()

    conn.close()

# ---------------- RESUME TEXT EXTRACTION ----------------

def extract_pdf_text(file_path):

    text = ""

    document = fitz.open(file_path)

    for page in document:

        text += page.get_text()

    document.close()

    return text

def extract_docx_text(file_path):

    document = Document(file_path)

    text = []

    for paragraph in document.paragraphs:

        text.append(paragraph.text)

    return "\n".join(text)

def extract_resume_text(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":

        return extract_pdf_text(file_path)

    if extension == ".docx":

        return extract_docx_text(file_path)

    return ""

# ---------------- SKILL EXTRACTION ----------------

SKILLS = [

    "python",

    "java",

    "c++",

    "c",

    "javascript",

    "typescript",

    "html",

    "css",

    "react",

    "angular",

    "node.js",

    "node",

    "express",

    "flask",

    "django",

    "spring boot",

    "sql",

    "mysql",

    "postgresql",

    "mongodb",

    "git",

    "github",

    "docker",

    "kubernetes",

    "aws",

    "azure",

    "machine learning",

    "deep learning",

    "data science",

    "data analysis",

    "pandas",

    "numpy",

    "scikit-learn",

    "tensorflow",

    "pytorch",

    "nlp",

    "artificial intelligence",

    "rest api",

    "api",

    "figma",

]

def extract_skills(text):

    text_lower = text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text_lower):

            found_skills.append(skill)

    return sorted(found_skills)

# ---------------- RESUME ANALYSIS ----------------

def analyze_resume(text):

    words = text.split()

    word_count = len(words)

    skills = extract_skills(text)

    # Email detection

    email_match = re.search(

        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",

        text

    )

    # Phone detection

    phone_match = re.search(
    r"(?:\+91[-\s]?)?[6-9]\d{9}",
    text
)

    email = (

        email_match.group(0)

        if email_match

        else "Not detected"

    )

    phone = (

        phone_match.group(0)

        if phone_match

        else "Not detected"

    )

    # ---------------- SECTION DETECTION ----------------

    sections = {

        "Education": bool(

            re.search(

                r"\b(education|academic|degree|btech|bachelor|master)\b",

                text,

                re.IGNORECASE

            )

        ),

        "Experience": bool(

            re.search(

                r"\b(experience|internship|employment|work experience)\b",

                text,

                re.IGNORECASE

            )

        ),

        "Projects": bool(

            re.search(

                r"\b(projects|project)\b",

                text,

                re.IGNORECASE

            )

        ),

        "Certifications": bool(

            re.search(

                r"\b(certification|certifications|certificate)\b",

                text,

                re.IGNORECASE

            )

        ),

        "Skills": len(skills) > 0

    }

    # ---------------- SCORE ----------------

    score = 0

    if word_count >= 150:

        score += 20

    if word_count >= 300:

        score += 10

    if email != "Not detected":

        score += 10

    if phone != "Not detected":

        score += 10

    score += min(len(skills) * 2, 25)

    score += sum(sections.values()) * 5

    score = min(score, 100)

    # ---------------- STRENGTHS ----------------

    strengths = []

    if email != "Not detected":

        strengths.append(

            "Email address detected"

        )

    if phone != "Not detected":

        strengths.append(

            "Phone number detected"

        )

    if len(skills) >= 5:

        strengths.append(

            "Good number of technical skills detected"

        )

    if sections["Projects"]:

        strengths.append(

            "Projects section is present"

        )

    if sections["Education"]:

        strengths.append(

            "Education section is present"

        )

    if sections["Experience"]:

        strengths.append(

            "Experience or internship section is present"

        )

    if sections["Certifications"]:

        strengths.append(

            "Certifications section is present"

        )

    if word_count >= 300:

        strengths.append(

            "Resume contains sufficient content"

        )

    # ---------------- SUGGESTIONS ----------------

    suggestions = []

    if email == "Not detected":

        suggestions.append(

            "Add a professional email address."

        )

    if phone == "Not detected":

        suggestions.append(

            "Add a valid phone number."

        )

    if word_count < 150:

        suggestions.append(

            "Resume content is too short. Add relevant details."

        )

    if not sections["Education"]:

        suggestions.append(

            "Add an Education section."

        )

    if not sections["Experience"]:

        suggestions.append(

            "Add internships, work experience, or relevant experience."

        )

    if not sections["Projects"]:

        suggestions.append(

            "Add 2–3 relevant projects with technologies used."

        )

    if not sections["Certifications"]:

        suggestions.append(

            "Add relevant certifications or courses."

        )

    if not sections["Skills"]:

        suggestions.append(

            "Add a dedicated Technical Skills section."

        )

    if len(skills) < 5:

        suggestions.append(

            "Add more relevant technical skills based on your target role."

        )

    if not suggestions:

        suggestions.append(

            "Your resume structure looks good. "

            "Focus on tailoring keywords for each job description."

        )

    return {

        "score": score,

        "word_count": word_count,

        "email": email,

        "phone": phone,

        "skills": skills,

        "sections": sections,

        "strengths": strengths,

        "suggestions": suggestions

    }

# ---------------- RESUME-JOB MATCHER ----------------

def match_resume_with_job(resume_text, job_text):

    resume_skills = set(

        extract_skills(resume_text)

    )

    job_skills = set(

        extract_skills(job_text)

    )

    matched_skills = sorted(

        resume_skills.intersection(job_skills)

    )

    missing_skills = sorted(

        job_skills - resume_skills

    )

    if not job_text.strip():

        similarity_score = 0

    else:

        vectorizer = TfidfVectorizer(

            stop_words="english"

        )

        vectors = vectorizer.fit_transform(

            [resume_text, job_text]

        )

        similarity_score = (

            cosine_similarity(

                vectors[0:1],

                vectors[1:2]

            )[0][0] * 100

        )

    skill_score = (

        (len(matched_skills) / len(job_skills)) * 100

        if job_skills

        else 0

    )

    final_match = round(

        (similarity_score * 0.6) +

        (skill_score * 0.4)

    )

    return {

        "match_score": min(

            final_match,

            100

        ),

        "similarity_score": round(

            similarity_score

        ),

        "skill_score": round(

            skill_score

        ),

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "job_skills": sorted(

            job_skills

        )

    }

# ---------------- HOME ----------------

@app.route("/")

def home():

    if "user_id" in session:

        return redirect(

            url_for("dashboard")

        )

    return redirect(

        url_for("login")

    )

# ---------------- REGISTER ----------------

@app.route("/register", methods=["GET", "POST"])

def register():

    if request.method == "POST":

        name = request.form["name"].strip()

        email = request.form["email"].strip().lower()

        password = request.form["password"]

        if not name or not email or not password:

            return "All fields are required."

        conn = get_db()

        try:

            hashed_password = generate_password_hash(

                password

            )

            conn.execute(

                """

                INSERT INTO users (name, email, password)

                VALUES (?, ?, ?)

                """,

                (

                    name,

                    email,

                    hashed_password

                )

            )

            conn.commit()

        except sqlite3.IntegrityError:

            conn.close()

            return "Email already registered."

        conn.close()

        return redirect(

            url_for("login")

        )

    return render_template(

        "register.html"

    )

# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET", "POST"])

def login():

    if request.method == "POST":

        email = request.form["email"].strip().lower()

        password = request.form["password"]

        conn = get_db()

        user = conn.execute(

            "SELECT * FROM users WHERE email = ?",

            (email,)

        ).fetchone()

        conn.close()

        if user and check_password_hash(

            user["password"],

            password

        ):

            session["user_id"] = user["id"]

            session["user_name"] = user["name"]

            return redirect(

                url_for("dashboard")

            )

        return "Invalid email or password."

    return render_template(

        "login.html"

    )

# ---------------- DASHBOARD ----------------

@app.route("/dashboard")

def dashboard():

    if "user_id" not in session:

        return redirect(

            url_for("login")

        )

    conn = get_db()

    # Get latest saved resume

    resume = conn.execute(

        """

        SELECT file_path

        FROM resumes

        WHERE user_id = ?

        ORDER BY uploaded_at DESC

        LIMIT 1

        """,

        (session["user_id"],)

    ).fetchone()

    # Get average job match

    match_data = conn.execute(

        """

        SELECT AVG(match_score) AS avg_match

        FROM job_matches

        WHERE user_id = ?

        """,

        (session["user_id"],)

    ).fetchone()

    # Get total applications

    application_data = conn.execute(

        """

        SELECT COUNT(*) AS total_applications

        FROM applications

        WHERE user_id = ?

        """,

        (session["user_id"],)

    ).fetchone()

    conn.close()

    resume_score = None

    skills_count = None

    avg_match = None

    applications_count = 0

    # Resume analysis

    if resume and os.path.exists(

        resume["file_path"]

    ):

        resume_text = extract_resume_text(

            resume["file_path"]

        )

        if resume_text.strip():

            analysis = analyze_resume(

                resume_text

            )

            resume_score = analysis["score"]

            skills_count = len(

                analysis["skills"]

            )

    # Average match score

    if (

        match_data

        and match_data["avg_match"] is not None

    ):

        avg_match = round(

            match_data["avg_match"]

        )

    # Application count

    if application_data:

        applications_count = application_data["total_applications"]

    return render_template(

        "dashboard.html",

        name=session["user_name"],

        resume_score=resume_score,

        skills_count=skills_count,

        avg_match=avg_match,

        applications_count=applications_count

    )

# ---------------- RESUME ANALYZER ----------------

@app.route("/resume", methods=["GET", "POST"])

def resume_analyzer():

    if "user_id" not in session:

        return redirect(

            url_for("login")

        )

    if request.method == "POST":

        file = request.files.get(

            "resume"

        )

        if not file or file.filename == "":

            return "Please select a resume."

        extension = os.path.splitext(

            file.filename

        )[1].lower()

        if extension not in [".pdf", ".docx"]:

            return "Only PDF and DOCX files are supported."

        safe_filename = (

            str(session["user_id"])

            + "_resume"

            + extension

        )

        file_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            safe_filename

        )

        file.save(file_path)

        # Save latest resume in database

        conn = get_db()

        conn.execute(

            "DELETE FROM resumes WHERE user_id = ?",

            (session["user_id"],)

        )

        conn.execute(

            """

            INSERT INTO resumes (user_id, file_path)

            VALUES (?, ?)

            """,

            (

                session["user_id"],

                file_path

            )

        )

        conn.commit()

        conn.close()

        text = extract_resume_text(

            file_path

        )

        if not text.strip():

            return "Could not extract text from the resume."

        analysis = analyze_resume(

            text

        )

        return render_template(

            "resume_result.html",

            analysis=analysis

        )

    return render_template(

        "resume.html"

    )

# ---------------- RESUME-JOB MATCHER ----------------

@app.route("/matcher", methods=["GET", "POST"])

def matcher():

    if "user_id" not in session:

        return redirect(

            url_for("login")

        )

    if request.method == "POST":

        job_text = request.form.get(

            "job_description",

            ""

        ).strip()

        if not job_text:

            return "Please enter the job description."

        # Get latest saved resume

        conn = get_db()

        resume = conn.execute(

            """

            SELECT file_path

            FROM resumes

            WHERE user_id = ?

            ORDER BY uploaded_at DESC

            LIMIT 1

            """,

            (session["user_id"],)

        ).fetchone()

        conn.close()

        if not resume:

            return "Please upload your resume in Resume Analyzer first."

        file_path = resume["file_path"]

        if not os.path.exists(file_path):

            return "Saved resume file not found. Please upload your resume again."

        resume_text = extract_resume_text(

            file_path

        )

        if not resume_text.strip():

            return "Could not extract text from the saved resume."

        result = match_resume_with_job(

            resume_text,

            job_text

        )

        # Save match score

        conn = get_db()

        conn.execute(

            """

            INSERT INTO job_matches (user_id, match_score)

            VALUES (?, ?)

            """,

            (

                session["user_id"],

                result["match_score"]

            )

        )

        conn.commit()

        conn.close()

        return render_template(

            "matcher_result.html",

            result=result

        )

    return render_template(

        "matcher.html"

    )

# ---------------- APPLICATION TRACKER ----------------

@app.route("/applications", methods=["GET", "POST"])

def applications():

    if "user_id" not in session:

        return redirect(

            url_for("login")

        )

    conn = get_db()

    if request.method == "POST":

        company = request.form.get(

            "company",

            ""

        ).strip()

        job_role = request.form.get(

            "job_role",

            ""

        ).strip()

        status = request.form.get(

            "status",

            "Saved"

        )

        applied_date = request.form.get(

            "applied_date",

            ""

        ).strip()

        if not company or not job_role:

            conn.close()

            return "Company name and job role are required."

        conn.execute(

            """

            INSERT INTO applications

            (user_id, company, job_role, status, applied_date)

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                session["user_id"],

                company,

                job_role,

                status,

                applied_date

            )

        )

        conn.commit()

    application_list = conn.execute(

        """

        SELECT id, company, job_role, status, applied_date

        FROM applications

        WHERE user_id = ?

        ORDER BY id DESC

        """,

        (session["user_id"],)

    ).fetchall()

    conn.close()

    return render_template(

        "application_tracker.html",

        applications=application_list

    )

# ---------------- LEARNING ROADMAP ----------------

@app.route("/roadmap", methods=["GET", "POST"])

def roadmap():

    if "user_id" not in session:

        return redirect(url_for("login"))

    roadmap_data = []

    message = None

    target_role = ""

    role_data = {

        "Software Developer": {

            "skills": [

                "python",

                "java",

                "sql",

                "git",

                "data structures",

                "algorithms"

            ]

        },

        "Frontend Developer": {

            "skills": [

                "html",

                "css",

                "javascript",

                "react",

                "git"

            ]

        },

        "Backend Developer": {

            "skills": [

                "python",

                "flask",

                "sql",

                "api",

                "docker",

                "git"

            ]

        },

        "Full Stack Developer": {

            "skills": [

                "html",

                "css",

                "javascript",

                "react",

                "node",

                "sql",

                "git"

            ]

        },

        "Data Analyst": {

            "skills": [

                "python",

                "sql",

                "pandas",

                "numpy",

                "data analysis",

                "excel"

            ]

        },

        "Data Scientist": {

            "skills": [

                "python",

                "sql",

                "pandas",

                "numpy",

                "statistics",

                "machine learning"

            ]

        },

        "Machine Learning Engineer": {

            "skills": [

                "python",

                "numpy",

                "pandas",

                "machine learning",

                "scikit-learn",

                "statistics"

            ]

        },

        "DevOps Engineer": {

            "skills": [

                "linux",

                "git",

                "docker",

                "ci/cd",

                "cloud",

                "kubernetes"

            ]

        },

        "Cloud Engineer": {

            "skills": [

                "linux",

                "networking",

                "cloud",

                "docker",

                "kubernetes",

                "git"

            ]

        }

    }

    learning_resources = {

        "python": {

            "level": "Beginner",

            "priority": "High",

            "duration": "4-6 weeks",

            "topics": [

                "Python fundamentals",

                "Functions and modules",

                "Object-oriented programming",

                "File handling",

                "Exception handling",

                "Libraries and virtual environments"

            ],

            "practice": "Build small Python programs and solve 30+ coding problems.",

            "project": "Build a command-line Student Management System."

        },

        "java": {

            "level": "Beginner",

            "priority": "High",

            "duration": "5-7 weeks",

            "topics": [

                "Java fundamentals",

                "Object-oriented programming",

                "Inheritance and polymorphism",

                "Collections Framework",

                "Exception handling",

                "JDBC"

            ],

            "practice": "Solve Java programming and OOP problems.",

            "project": "Build a Java-based Library Management System."

        },

        "sql": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "3-4 weeks",

            "topics": [

                "SELECT and filtering",

                "JOINs",

                "GROUP BY and HAVING",

                "Subqueries",

                "Indexes",

                "Window functions"

            ],

            "practice": "Solve 40+ SQL queries using a real-world dataset.",

            "project": "Build a Job Application Analytics Database."

        },

        "git": {

            "level": "Beginner",

            "priority": "Medium",

            "duration": "1-2 weeks",

            "topics": [

                "Repositories",

                "Commit and push",

                "Branches",

                "Merge and pull requests",

                "GitHub workflow"

            ],

            "practice": "Use Git for every project.",

            "project": "Create and maintain a professional GitHub portfolio."

        },

        "html": {

            "level": "Beginner",

            "priority": "High",

            "duration": "1-2 weeks",

            "topics": [

                "HTML structure",

                "Forms",

                "Tables",

                "Semantic HTML",

                "Accessibility"

            ],

            "practice": "Build responsive page structures.",

            "project": "Build a professional portfolio website."

        },

        "css": {

            "level": "Beginner",

            "priority": "High",

            "duration": "2-3 weeks",

            "topics": [

                "Selectors",

                "Box model",

                "Flexbox",

                "CSS Grid",

                "Responsive design",

                "Animations"

            ],

            "practice": "Recreate 3 modern website layouts.",

            "project": "Build a responsive portfolio dashboard."

        },

        "javascript": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "4-6 weeks",

            "topics": [

                "JavaScript fundamentals",

                "DOM manipulation",

                "Events",

                "Promises",

                "Async/Await",

                "Fetch API"

            ],

            "practice": "Build interactive browser applications.",

            "project": "Build a Job Search Dashboard using APIs."

        },

        "react": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "4-6 weeks",

            "topics": [

                "Components",

                "Props and state",

                "Hooks",

                "Routing",

                "API integration",

                "State management"

            ],

            "practice": "Build reusable React components.",

            "project": "Build a React-based Career Dashboard."

        },

        "flask": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "3-4 weeks",

            "topics": [

                "Flask application structure",

                "Routing",

                "Templates",

                "Forms",

                "Database integration",

                "REST APIs"

            ],

            "practice": "Build REST endpoints and database-driven applications.",

            "project": "Build a Resume Management API."

        },

        "api": {

            "level": "Intermediate",

            "priority": "Medium",

            "duration": "2-3 weeks",

            "topics": [

                "HTTP methods",

                "REST architecture",

                "JSON",

                "Authentication",

                "API requests",

                "Error handling"

            ],

            "practice": "Consume and create REST APIs.",

            "project": "Build a Job Listing REST API."

        },

        "docker": {

            "level": "Intermediate",

            "priority": "Medium",

            "duration": "2-3 weeks",

            "topics": [

                "Images",

                "Containers",

                "Dockerfile",

                "Volumes",

                "Networking",

                "Docker Compose"

            ],

            "practice": "Containerize existing projects.",

            "project": "Dockerize a Flask and MySQL application."

        },

        "node": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "3-4 weeks",

            "topics": [

                "Node.js fundamentals",

                "NPM",

                "Express.js",

                "REST APIs",

                "Middleware",

                "Database connectivity"

            ],

            "practice": "Build backend APIs with Express.",

            "project": "Build a Job Portal Backend."

        },

        "pandas": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "2-3 weeks",

            "topics": [

                "DataFrames",

                "Data cleaning",

                "Filtering",

                "Grouping",

                "Merging datasets",

                "Data analysis"

            ],

            "practice": "Analyze real-world datasets.",

            "project": "Build an Employee Data Analysis Dashboard."

        },

        "numpy": {

            "level": "Beginner",

            "priority": "Medium",

            "duration": "1-2 weeks",

            "topics": [

                "Arrays",

                "Indexing",

                "Array operations",

                "Mathematical functions",

                "Broadcasting"

            ],

            "practice": "Solve numerical computing exercises.",

            "project": "Build a basic data-processing application."

        },

        "data analysis": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "3-4 weeks",

            "topics": [

                "Data cleaning",

                "Exploratory data analysis",

                "Data visualization",

                "Statistical analysis",

                "Business insights"

            ],

            "practice": "Analyze real datasets and write insights.",

            "project": "Build a Sales Analytics Dashboard."

        },

        "excel": {

            "level": "Beginner",

            "priority": "Medium",

            "duration": "2-3 weeks",

            "topics": [

                "Formulas",

                "Functions",

                "Sorting and filtering",

                "Pivot tables",

                "Charts",

                "Data dashboards"

            ],

            "practice": "Work with business datasets.",

            "project": "Build an Excel-based Business Dashboard."

        },

        "machine learning": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "6-8 weeks",

            "topics": [

                "Supervised learning",

                "Unsupervised learning",

                "Regression",

                "Classification",

                "Clustering",

                "Model evaluation"

            ],

            "practice": "Train and evaluate models on real datasets.",

            "project": "Build an ML-based prediction system."

        },

        "scikit-learn": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "3-4 weeks",

            "topics": [

                "Data preprocessing",

                "Train-test split",

                "Classification",

                "Regression",

                "Model evaluation",

                "Pipelines"

            ],

            "practice": "Implement multiple ML models.",

            "project": "Build a Resume Classification System."

        },

        "statistics": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "3-4 weeks",

            "topics": [

                "Mean and median",

                "Probability",

                "Distributions",

                "Variance and standard deviation",

                "Hypothesis testing",

                "Correlation"

            ],

            "practice": "Solve statistical problems using datasets.",

            "project": "Perform statistical analysis on a real dataset."

        },

        "data structures": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "6-8 weeks",

            "topics": [

                "Arrays",

                "Strings",

                "Linked lists",

                "Stacks and queues",

                "Trees",

                "Graphs",

                "Hashing"

            ],

            "practice": "Solve 50+ DSA problems.",

            "project": "Build a DSA problem-solving application."

        },

        "algorithms": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "5-7 weeks",

            "topics": [

                "Searching",

                "Sorting",

                "Recursion",

                "Greedy algorithms",

                "Divide and conquer",

                "Dynamic programming"

            ],

            "practice": "Solve algorithmic problems on coding platforms.",

            "project": "Build an Algorithm Visualizer."

        },

        "linux": {

            "level": "Beginner",

            "priority": "High",

            "duration": "2-3 weeks",

            "topics": [

                "Linux commands",

                "File permissions",

                "Processes",

                "Shell scripting",

                "Package management"

            ],

            "practice": "Use Linux for development tasks.",

            "project": "Create shell scripts for system automation."

        },

        "ci/cd": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "2-3 weeks",

            "topics": [

                "Continuous integration",

                "Continuous deployment",

                "GitHub Actions",

                "Build pipelines",

                "Automated testing"

            ],

            "practice": "Create CI/CD pipelines for projects.",

            "project": "Automate deployment of a web application."

        },

        "cloud": {

            "level": "Intermediate",

            "priority": "High",

            "duration": "4-6 weeks",

            "topics": [

                "Cloud fundamentals",

                "Compute",

                "Storage",

                "Networking",

                "IAM",

                "Deployment"

            ],

            "practice": "Deploy applications to a cloud platform.",

            "project": "Deploy a full-stack application to the cloud."

        },

        "kubernetes": {

            "level": "Advanced",

            "priority": "Medium",

            "duration": "4-6 weeks",

            "topics": [

                "Pods",

                "Deployments",

                "Services",

                "ConfigMaps",

                "Secrets",

                "Scaling"

            ],

            "practice": "Deploy containerized applications.",

            "project": "Deploy a microservice application using Kubernetes."

        },

        "networking": {

            "level": "Beginner",

            "priority": "High",

            "duration": "3-4 weeks",

            "topics": [

                "OSI model",

                "TCP/IP",

                "IP addressing",

                "DNS",

                "HTTP/HTTPS",

                "Routing"

            ],

            "practice": "Practice networking commands and concepts.",

            "project": "Build a basic network monitoring tool."

        }

    }

    if request.method == "POST":

        target_role = request.form.get(

            "target_role",

            ""

        ).strip()

        if not target_role:

            message = "Please select a target job role."

        elif target_role not in role_data:

            message = "Invalid target role selected."

        else:

            conn = get_db()

            resume = conn.execute(

                """

                SELECT file_path

                FROM resumes

                WHERE user_id = ?

                ORDER BY uploaded_at DESC

                LIMIT 1

                """,

                (session["user_id"],)

            ).fetchone()

            conn.close()

            if not resume:

                message = "Please upload your resume first."

            else:

                file_path = resume["file_path"]

                if not os.path.exists(file_path):

                    message = (

                        "Saved resume not found. "

                        "Please upload your resume again."

                    )

                else:

                    resume_text = extract_resume_text(

                        file_path

                    )

                    if not resume_text.strip():

                        message = (

                            "Could not extract text from your resume."

                        )

                    else:

                        user_skills = set(

                            extract_skills(resume_text)

                        )

                        required_skills = set(

                            role_data[target_role]["skills"]

                        )

                        missing_skills = (

                            required_skills - user_skills

                        )

                        for skill in sorted(

                            missing_skills

                        ):

                            resource = learning_resources.get(

                                skill,

                                {

                                    "level": "Beginner",

                                    "priority": "Medium",

                                    "duration": "2-3 weeks",

                                    "topics": [

                                        "Learn the fundamentals",

                                        "Practice important concepts",

                                        "Build a small project"

                                    ],

                                    "practice": (

                                        "Practice with "

                                        "real-world problems."

                                    ),

                                    "project": (

                                        "Build a small "

                                        "portfolio project."

                                    )

                                }

                            )

                            roadmap_data.append(

                                {

                                    "skill": skill,

                                    "level": resource["level"],

                                    "priority": resource["priority"],

                                    "duration": resource["duration"],

                                    "topics": resource["topics"],

                                    "practice": resource["practice"],

                                    "project": resource["project"]

                                }

                            )

                        if not roadmap_data:

                            message = (

                                "Your resume already covers "

                                "the core skills for this role."

                            )

    return render_template(

        "roadmap.html",

        roadmap=roadmap_data,

        message=message,

        target_role=target_role,

        roles=role_data.keys()

    )

# ---------------- SKILL GAP ANALYZER ----------------

@app.route("/skill-gap", methods=["GET", "POST"])
def skill_gap():

    if "user_id" not in session:
        return redirect(url_for("login"))

    result = None
    message = None

    role_data = {

        "Software Developer": {
            "skills": [
                "python",
                "java",
                "sql",
                "git",
                "data structures",
                "algorithms"
            ],
            "categories": {
                "Programming": ["python", "java"],
                "Database": ["sql"],
                "Development Tools": ["git"],
                "Computer Science": [
                    "data structures",
                    "algorithms"
                ]
            }
        },

        "Frontend Developer": {
            "skills": [
                "html",
                "css",
                "javascript",
                "react",
                "git"
            ],
            "categories": {
                "Frontend": [
                    "html",
                    "css",
                    "javascript",
                    "react"
                ],
                "Development Tools": ["git"]
            }
        },

        "Backend Developer": {
            "skills": [
                "python",
                "flask",
                "sql",
                "api",
                "docker",
                "git"
            ],
            "categories": {
                "Programming": ["python"],
                "Backend": ["flask", "api"],
                "Database": ["sql"],
                "DevOps": ["docker"],
                "Development Tools": ["git"]
            }
        },

        "Data Analyst": {
            "skills": [
                "python",
                "sql",
                "excel",
                "statistics",
                "pandas",
                "data visualization"
            ],
            "categories": {
                "Programming": ["python"],
                "Database": ["sql"],
                "Tools": ["excel", "pandas"],
                "Analytics": [
                    "statistics",
                    "data visualization"
                ]
            }
        },

        "Machine Learning Engineer": {
            "skills": [
                "python",
                "machine learning",
                "numpy",
                "pandas",
                "scikit-learn",
                "statistics"
            ],
            "categories": {
                "Programming": ["python"],
                "Machine Learning": [
                    "machine learning",
                    "scikit-learn"
                ],
                "Data Libraries": [
                    "numpy",
                    "pandas"
                ],
                "Mathematics": ["statistics"]
            }
        },

        "Full Stack Developer": {
            "skills": [
                "html",
                "css",
                "javascript",
                "react",
                "python",
                "flask",
                "sql",
                "git"
            ],
            "categories": {
                "Frontend": [
                    "html",
                    "css",
                    "javascript",
                    "react"
                ],
                "Backend": [
                    "python",
                    "flask"
                ],
                "Database": ["sql"],
                "Development Tools": ["git"]
            }
        }
    }

    if request.method == "POST":

        target_role = request.form.get(
            "target_role",
            ""
        ).strip()

        if target_role not in role_data:

            message = "Please select a valid target role."

        else:

            conn = get_db()

            resume = conn.execute(
                """
                SELECT file_path
                FROM resumes
                WHERE user_id = ?
                ORDER BY uploaded_at DESC
                LIMIT 1
                """,
                (session["user_id"],)
            ).fetchone()

            conn.close()

            if not resume:

                message = (
                    "Please upload your resume in Resume Analyzer first."
                )

            else:

                file_path = resume["file_path"]

                if not os.path.exists(file_path):

                    message = (
                        "Saved resume file not found. "
                        "Please upload your resume again."
                    )

                else:

                    try:

                        resume_text = extract_resume_text(
                            file_path
                        )

                        if not resume_text.strip():

                            message = (
                                "Could not extract text from the resume."
                            )

                        else:

                            resume_text = resume_text.lower()

                            user_skills = extract_skills(
                                resume_text
                            )

                            required_skills = role_data[
                                target_role
                            ]["skills"]

                            matched_skills = []
                            missing_skills = []

                            for skill in required_skills:

                                if skill.lower() in user_skills:
                                    matched_skills.append(skill)

                                else:
                                    missing_skills.append(skill)

                            total_required = len(
                                required_skills
                            )

                            match_percentage = round(
                                (
                                    len(matched_skills)
                                    / total_required
                                ) * 100
                            ) if total_required else 0

                            skill_gap_percentage = (
                                100 - match_percentage
                            )

                            # ---------------- READINESS ----------------

                            if match_percentage >= 85:
                                readiness = "Highly Ready"
                                readiness_text = (
                                    "Your resume covers most of the "
                                    "important skills for this role."
                                )

                            elif match_percentage >= 65:
                                readiness = "Nearly Ready"
                                readiness_text = (
                                    "You have a good foundation, but "
                                    "some important skills are still missing."
                                )

                            elif match_percentage >= 40:
                                readiness = "Developing"
                                readiness_text = (
                                    "You have started building the required "
                                    "skill set. Focus on the missing skills."
                                )

                            else:
                                readiness = "Early Stage"
                                readiness_text = (
                                    "Your current skill set has a significant "
                                    "gap for this target role."
                                )

                            # ---------------- PRIORITY ----------------

                            high_priority = []
                            medium_priority = []
                            low_priority = []

                            for skill in missing_skills:

                                if skill in [
                                    "python",
                                    "java",
                                    "javascript",
                                    "sql",
                                    "data structures",
                                    "algorithms",
                                    "machine learning",
                                    "react"
                                ]:
                                    high_priority.append(skill)

                                elif skill in [
                                    "git",
                                    "flask",
                                    "api",
                                    "docker",
                                    "pandas",
                                    "numpy",
                                    "scikit-learn",
                                    "statistics"
                                ]:
                                    medium_priority.append(skill)

                                else:
                                    low_priority.append(skill)

                            # ---------------- CATEGORY ANALYSIS ----------------

                            category_data = []

                            for category, skills in role_data[
                                target_role
                            ]["categories"].items():

                                category_matched = 0

                                for skill in skills:

                                    if skill in user_skills:
                                        category_matched += 1

                                category_percentage = round(
                                    (
                                        category_matched
                                        / len(skills)
                                    ) * 100
                                ) if skills else 0

                                category_data.append({
                                    "name": category,
                                    "percentage": category_percentage,
                                    "matched": category_matched,
                                    "total": len(skills)
                                })

                            # ---------------- LEARNING PLAN ----------------

                            learning_map = {

                                "python":
                                    "Learn Python fundamentals, OOP, functions, modules and problem solving.",

                                "java":
                                    "Focus on Java OOP, collections, exception handling and backend development.",

                                "sql":
                                    "Practice SQL queries, joins, subqueries, aggregation and database design.",

                                "git":
                                    "Learn Git workflow, branching, commits, merging and GitHub collaboration.",

                                "data structures":
                                    "Practice arrays, linked lists, stacks, queues, trees, graphs and hash tables.",

                                "algorithms":
                                    "Practice searching, sorting, recursion, greedy and dynamic programming.",

                                "html":
                                    "Learn semantic HTML and build structured responsive web pages.",

                                "css":
                                    "Practice Flexbox, Grid, responsive layouts and modern UI styling.",

                                "javascript":
                                    "Learn JavaScript fundamentals, DOM, events, ES6 and asynchronous programming.",

                                "react":
                                    "Learn React components, props, state, hooks and API integration.",

                                "flask":
                                    "Build REST APIs and backend applications using Flask.",

                                "api":
                                    "Learn REST APIs, HTTP methods, JSON, authentication and API integration.",

                                "docker":
                                    "Learn containers, Dockerfiles, images and application deployment.",

                                "excel":
                                    "Practice formulas, pivot tables, charts and data cleaning in Excel.",

                                "statistics":
                                    "Study descriptive statistics, probability, distributions and hypothesis testing.",

                                "pandas":
                                    "Practice data cleaning, filtering, grouping and analysis with Pandas.",

                                "data visualization":
                                    "Learn charts, dashboards and data visualization using Python libraries.",

                                "machine learning":
                                    "Learn supervised learning, unsupervised learning and model evaluation.",

                                "numpy":
                                    "Practice arrays, mathematical operations and numerical computing with NumPy.",

                                "scikit-learn":
                                    "Learn preprocessing, model training, evaluation and ML pipelines."
                            }

                            learning_plan = []

                            for skill in missing_skills:

                                learning_plan.append({
                                    "skill": skill,
                                    "description": learning_map.get(
                                        skill,
                                        "Build practical projects and practice this skill regularly."
                                    )
                                })

                            result = {

                                "target_role": target_role,

                                "match_percentage":
                                    match_percentage,

                                "skill_gap":
                                    skill_gap_percentage,

                                "user_skills":
                                    user_skills,

                                "matched_skills":
                                    matched_skills,

                                "missing_skills":
                                    missing_skills,

                                "required_skills":
                                    required_skills,

                                "readiness":
                                    readiness,

                                "readiness_text":
                                    readiness_text,

                                "high_priority":
                                    high_priority,

                                "medium_priority":
                                    medium_priority,

                                "low_priority":
                                    low_priority,

                                "category_data":
                                    category_data,

                                "learning_plan":
                                    learning_plan,

                                "total_required":
                                    total_required,

                                "total_matched":
                                    len(matched_skills),

                                "total_missing":
                                    len(missing_skills)
                            }

                    except Exception as e:

                        print(
                            "SKILL GAP ERROR:",
                            str(e)
                        )

                        message = (
                            "Error while analyzing your resume: "
                            + str(e)
                        )

    return render_template(
        "skill_gap.html",
        result=result,
        message=message,
        roles=list(role_data.keys())
    )
# ---------------- JOB RECOMMENDATIONS ----------------

@app.route("/recommendations")

def recommendations():

    if "user_id" not in session:

        return redirect(url_for("login"))

    conn = get_db()

    resume = conn.execute(

        """

        SELECT file_path

        FROM resumes

        WHERE user_id = ?

        ORDER BY uploaded_at DESC

        LIMIT 1

        """,

        (session["user_id"],)

    ).fetchone()

    conn.close()

    if not resume:

        return render_template(

            "recommendations.html",

            recommendations=[],

            message="Please upload your resume first."

        )

    file_path = resume["file_path"]

    if not os.path.exists(file_path):

        return render_template(

            "recommendations.html",

            recommendations=[],

            message="Saved resume not found. Please upload your resume again."

        )

    resume_text = extract_resume_text(file_path)

    if not resume_text.strip():

        return render_template(

            "recommendations.html",

            recommendations=[],

            message="Could not extract text from your resume."

        )

    resume_skills = set(extract_skills(resume_text))

    # Sample job database

    jobs = [

        {

            "title": "Software Developer",

            "company": "Tech Solutions",

            "skills": ["python", "java", "sql", "git"],

            "description": "Develop and maintain software applications."

        },

        {

            "title": "Frontend Developer",

            "company": "WebTech",

            "skills": ["html", "css", "javascript", "react"],

            "description": "Build responsive and interactive web applications."

        },

        {

            "title": "Backend Developer",

            "company": "Cloud Systems",

            "skills": ["python", "flask", "sql", "api", "docker"],

            "description": "Develop backend services and REST APIs."

        },

        {

            "title": "Data Analyst",

            "company": "DataWorks",

            "skills": ["python", "sql", "pandas", "numpy", "data analysis"],

            "description": "Analyze data and generate useful business insights."

        },

        {

            "title": "Machine Learning Engineer",

            "company": "AI Labs",

            "skills": ["python", "machine learning", "numpy",

                       "pandas", "scikit-learn"],

            "description": "Build and evaluate machine learning models."

        },

        {

            "title": "Full Stack Developer",

            "company": "Digital Solutions",

            "skills": ["html", "css", "javascript", "react",

                       "node", "sql", "git"],

            "description": "Develop complete frontend and backend web applications."

        }

    ]

    recommendations = []

    for job in jobs:

        job_skills = set(job["skills"])

        matched = resume_skills.intersection(job_skills)

        score = round(

            (len(matched) / len(job_skills)) * 100

        )

        recommendations.append({

            "title": job["title"],

            "company": job["company"],

            "description": job["description"],

            "match_score": score,

            "matched_skills": sorted(matched),

            "missing_skills": sorted(

                job_skills - resume_skills

            )

        })

    # Highest matching jobs first

    recommendations.sort(

        key=lambda x: x["match_score"],

        reverse=True

    )

    return render_template(

        "recommendations.html",

        recommendations=recommendations,

        message=None

    )

# ---------------- ATS CHECKER ----------------

@app.route("/ats-checker", methods=["GET", "POST"])
def ats_checker():

    if "user_id" not in session:
        return redirect(url_for("login"))

    result = None
    message = None

    if request.method == "POST":

        job_description = request.form.get("job_description", "").strip()

        conn = get_db()

        resume = conn.execute(
            """
            SELECT file_path
            FROM resumes
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
            LIMIT 1
            """,
            (session["user_id"],)
        ).fetchone()

        conn.close()

        if not resume:

            message = "Please upload your resume in Resume Analyzer first."

        else:

            file_path = resume["file_path"]

            if not os.path.exists(file_path):

                message = (
                    "Saved resume file not found. "
                    "Please upload your resume again."
                )

            else:

                try:

                    resume_text = extract_resume_text(file_path)

                    if not resume_text.strip():

                        message = "Could not extract text from the resume."

                    else:

                        text = resume_text.lower()

                        # --------------------------------
                        # BASIC DATA
                        # --------------------------------

                        word_count = len(resume_text.split())

                        suggestions = []
                        warnings = []

                        # --------------------------------
                        # CONTACT INFORMATION
                        # --------------------------------

                        email_found = bool(
                            re.search(
                                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                                resume_text
                            )
                        )

                        phone_pattern = r"(?:\+91[-\s]?)?[6-9]\d{9}"

                        phone_found = bool(
                            re.search(
                                phone_pattern,
                                resume_text
                            )
                        )

                        linkedin_found = "linkedin.com" in text
                        github_found = "github.com" in text

                        contact_score = 0

                        if email_found:
                            contact_score += 5
                        else:
                            suggestions.append(
                                "Add a professional email address."
                            )

                        if phone_found:
                            contact_score += 5
                        else:
                            suggestions.append(
                                "Add a valid phone number."
                            )

                        if linkedin_found:
                            contact_score += 2

                        if github_found:
                            contact_score += 2

                        # --------------------------------
                        # RESUME SECTIONS
                        # --------------------------------

                        section_keywords = {

                            "Summary": [
                                "summary",
                                "profile",
                                "objective"
                            ],

                            "Education": [
                                "education",
                                "academic"
                            ],

                            "Experience": [
                                "experience",
                                "work experience",
                                "internship"
                            ],

                            "Skills": [
                                "skills",
                                "technical skills"
                            ],

                            "Projects": [
                                "projects",
                                "project"
                            ],

                            "Certifications": [
                                "certification",
                                "certifications",
                                "certificate"
                            ]
                        }

                        section_status = {}
                        sections_found = 0

                        for section, keywords in section_keywords.items():

                            found = False

                            for keyword in keywords:

                                if keyword in text:
                                    found = True
                                    break

                            section_status[section] = found

                            if found:

                                sections_found += 1

                            else:

                                if section in [
                                    "Summary",
                                    "Education",
                                    "Skills",
                                    "Projects"
                                ]:

                                    suggestions.append(
                                        f"Add a clear {section} section."
                                    )

                        section_score = round(
                            (sections_found / len(section_keywords)) * 25
                        )

                        # --------------------------------
                        # TECHNICAL SKILLS
                        # --------------------------------

                        technical_keywords = [

                            "python",
                            "java",
                            "c++",
                            "c",
                            "javascript",
                            "typescript",
                            "sql",
                            "html",
                            "css",
                            "git",
                            "github",
                            "flask",
                            "django",
                            "react",
                            "node.js",
                            "node",
                            "express",
                            "mongodb",
                            "mysql",
                            "postgresql",
                            "machine learning",
                            "deep learning",
                            "artificial intelligence",
                            "numpy",
                            "pandas",
                            "scikit-learn",
                            "tensorflow",
                            "pytorch",
                            "data structures",
                            "algorithms",
                            "oop",
                            "rest api",
                            "api",
                            "docker",
                            "linux",
                            "aws",
                            "azure"
                        ]

                        detected_keywords = []

                        for keyword in technical_keywords:

                            if keyword in text:

                                detected_keywords.append(
                                    keyword
                                )

                        keyword_score = min(
                            round(
                                (len(detected_keywords) / 12) * 20
                            ),
                            20
                        )

                        if len(detected_keywords) < 5:

                            suggestions.append(
                                "Add more relevant technical skills and keywords."
                            )

                        # --------------------------------
                        # ACTION VERBS
                        # --------------------------------

                        action_verbs = [

                            "developed",
                            "built",
                            "designed",
                            "implemented",
                            "created",
                            "optimized",
                            "automated",
                            "integrated",
                            "managed",
                            "tested",
                            "deployed",
                            "analyzed",
                            "engineered",
                            "configured",
                            "maintained",
                            "improved",
                            "led",
                            "collaborated",
                            "solved"
                        ]

                        detected_action_verbs = []

                        for verb in action_verbs:

                            if re.search(
                                r"\b" + re.escape(verb) + r"\b",
                                text
                            ):

                                detected_action_verbs.append(
                                    verb
                                )

                        action_verb_score = min(
                            round(
                                (len(detected_action_verbs) / 6) * 10
                            ),
                            10
                        )

                        if len(detected_action_verbs) < 3:

                            suggestions.append(
                                "Use stronger action verbs such as "
                                "developed, implemented, designed or optimized."
                            )

                        # --------------------------------
                        # RESUME LENGTH
                        # --------------------------------

                        length_score = 0
                        length_status = ""

                        if word_count < 250:

                            length_status = "Too Short"

                            warnings.append(
                                "Resume contains very little content."
                            )

                            suggestions.append(
                                "Add relevant projects, technical skills, "
                                "education or experience."
                            )

                        elif word_count <= 900:

                            length_status = "Good"
                            length_score = 10

                        else:

                            length_status = "Long"

                            suggestions.append(
                                "Keep the resume concise and remove "
                                "unnecessary information."
                            )

                        # --------------------------------
                        # FORMATTING / ATS READABILITY
                        # --------------------------------

                        formatting_score = 0

                        if len(resume_text.strip()) > 100:
                            formatting_score += 3

                        if "\x00" not in resume_text:
                            formatting_score += 2

                        if resume_text.count("\n") >= 5:
                            formatting_score += 2

                        if email_found:
                            formatting_score += 1

                        if phone_found:
                            formatting_score += 1

                        if sections_found >= 4:
                            formatting_score += 1

                        formatting_score = min(
                            formatting_score,
                            10
                        )

                        if formatting_score < 6:

                            warnings.append(
                                "Resume text structure may not be fully ATS-friendly."
                            )

                            suggestions.append(
                                "Use clear section headings, simple formatting "
                                "and readable text."
                            )

                        # --------------------------------
                        # COMMON KEYWORDS
                        # --------------------------------

                        common_keywords = [

                            "problem solving",
                            "communication",
                            "teamwork",
                            "leadership",
                            "debugging",
                            "testing",
                            "version control",
                            "database",
                            "api",
                            "software development"
                        ]

                        missing_keywords = []

                        for keyword in common_keywords:

                            if keyword not in text:

                                missing_keywords.append(
                                    keyword
                                )

                        # --------------------------------
                        # GENERAL ATS SCORE
                        # --------------------------------

                        raw_score = (

                            contact_score
                            + section_score
                            + keyword_score
                            + action_verb_score
                            + length_score
                            + formatting_score

                        )

                        # Current components total maximum = 89.
                        # Convert it properly to a score out of 100.

                        score = round(
                            (raw_score / 89) * 100
                        )

                        score = min(
                            score,
                            100
                        )

                        # --------------------------------
                        # SCORE LABEL
                        # --------------------------------

                        if score >= 80:

                            score_label = "Strong ATS Compatibility"

                        elif score >= 60:

                            score_label = "Good ATS Compatibility"

                        elif score >= 40:

                            score_label = "Needs Improvement"

                        else:

                            score_label = "Low ATS Compatibility"

                        # --------------------------------
                        # JOB DESCRIPTION MATCHING
                        # --------------------------------

                        job_description_provided = bool(
                            job_description
                        )

                        job_match_score = None
                        matched_job_keywords = []
                        missing_job_keywords = []
                        job_match_label = ""

                        if job_description_provided:

                            job_text = job_description.lower()

                            # Words/phrases useful for job matching
                            job_keywords = [

                                "python",
                                "java",
                                "c++",
                                "javascript",
                                "typescript",
                                "html",
                                "css",
                                "react",
                                "node.js",
                                "node",
                                "express",
                                "flask",
                                "django",
                                "sql",
                                "mysql",
                                "postgresql",
                                "mongodb",
                                "git",
                                "github",
                                "docker",
                                "aws",
                                "azure",
                                "linux",
                                "api",
                                "rest api",
                                "machine learning",
                                "deep learning",
                                "artificial intelligence",
                                "numpy",
                                "pandas",
                                "scikit-learn",
                                "tensorflow",
                                "pytorch",
                                "data structures",
                                "algorithms",
                                "object oriented programming",
                                "oop",
                                "database",
                                "testing",
                                "debugging",
                                "problem solving",
                                "communication",
                                "teamwork",
                                "leadership",
                                "software development",
                                "web development",
                                "backend",
                                "frontend",
                                "full stack",
                                "data analysis"
                            ]

                            required_job_keywords = []

                            for keyword in job_keywords:

                                if keyword in job_text:

                                    required_job_keywords.append(
                                        keyword
                                    )

                            for keyword in required_job_keywords:

                                if keyword in text:

                                    matched_job_keywords.append(
                                        keyword
                                    )

                                else:

                                    missing_job_keywords.append(
                                        keyword
                                    )

                            if len(required_job_keywords) > 0:

                                job_match_score = round(
                                    (
                                        len(matched_job_keywords)
                                        / len(required_job_keywords)
                                    ) * 100
                                )

                            else:

                                job_match_score = 0

                            if job_match_score >= 80:

                                job_match_label = "Excellent Job Match"

                            elif job_match_score >= 60:

                                job_match_label = "Good Job Match"

                            elif job_match_score >= 40:

                                job_match_label = "Partial Job Match"

                            else:

                                job_match_label = "Low Job Match"

                        # --------------------------------
                        # FINAL RESULT
                        # --------------------------------

                        result = {

                            "score": score,

                            "score_label": score_label,

                            "word_count": word_count,

                            "length_status": length_status,

                            "detected_keywords":
                                detected_keywords,

                            "missing_keywords":
                                missing_keywords,

                            "detected_action_verbs":
                                detected_action_verbs,

                            "sections_found":
                                sections_found,

                            "total_sections":
                                len(section_keywords),

                            "section_status":
                                section_status,

                            "email_found":
                                email_found,

                            "phone_found":
                                phone_found,

                            "linkedin_found":
                                linkedin_found,

                            "github_found":
                                github_found,

                            "contact_score":
                                contact_score,

                            "section_score":
                                section_score,

                            "keyword_score":
                                keyword_score,

                            "action_verb_score":
                                action_verb_score,

                            "length_score":
                                length_score,

                            "formatting_score":
                                formatting_score,

                            "formatting_status":
                                "Good"
                                if formatting_score >= 6
                                else "Needs Attention",

                            "suggestions":
                                suggestions,

                            "warnings":
                                warnings,

                            # Job matching
                            "job_description_provided":
                                job_description_provided,

                            "job_match_score":
                                job_match_score,

                            "matched_job_keywords":
                                matched_job_keywords,

                            "missing_job_keywords":
                                missing_job_keywords,

                            "job_match_label":
                                job_match_label
                        }

                except Exception as e:

                    message = (
                        "Error while checking your resume: "
                        + str(e)
                    )

    return render_template(
        "ats_checker.html",
        result=result,
        message=message
    )

# ---------------- INTERVIEW QUESTION BANK ----------------

INTERVIEW_QUESTIONS = {

    "Software Developer": {
        "Technical": [
            "Tell me about your Hospital Management System project.",
            "Why did you choose Java for your project?",
            "What is the difference between an Array and an ArrayList?",
            "Explain the four pillars of Object-Oriented Programming.",
            "What is the difference between method overloading and method overriding?",
            "What is a primary key and why is it important in a database?",
            "Explain the difference between INNER JOIN and LEFT JOIN.",
            "What is normalization in DBMS?",
            "What is the difference between a process and a thread?",
            "Explain what happens when you enter a URL in a web browser.",
            "What is the difference between HTTP and HTTPS?",
            "What is Git and why is it used in software development?",
            "How would you debug a program that suddenly stops working?",
            "How would you improve the performance of a slow application?",
            "If your application had 10,000 users, what changes would you consider?"
        ],

        "HR": [
            "Tell me about yourself.",
            "Why do you want to become a software developer?",
            "What are your strengths?",
            "What is one technical skill you are currently improving?",
            "Tell me about a difficult problem you faced in a project.",
            "How do you handle deadlines?",
            "How do you handle criticism?",
            "Where do you see yourself in the next five years?"
        ],

        "Behavioral": [
            "Tell me about a time when you faced a difficult technical problem.",
            "Describe a situation where you had to learn something quickly.",
            "Tell me about a mistake you made in a project and what you learned from it.",
            "Describe a time when you had to work under pressure.",
            "Tell me about a situation where you disagreed with a teammate.",
            "How do you prioritize multiple tasks?",
            "Describe a project where you took responsibility for an important task."
        ]
    },


    "Frontend Developer": {
        "Technical": [
            "Explain the difference between HTML, CSS and JavaScript.",
            "What is the DOM?",
            "What is the difference between let, const and var?",
            "Explain event bubbling in JavaScript.",
            "What is responsive web design?",
            "What is the difference between Flexbox and CSS Grid?",
            "What are React components?",
            "What is the difference between state and props in React?",
            "What is the Virtual DOM?",
            "How would you improve the performance of a frontend application?"
        ],

        "HR": [
            "Tell me about yourself.",
            "Why did you choose frontend development?",
            "What type of websites do you enjoy building?",
            "How do you handle feedback on your UI design?"
        ],

        "Behavioral": [
            "Tell me about a difficult UI problem you solved.",
            "Describe a time when you had to learn a new frontend technology.",
            "Tell me about a project where you worked with a team."
        ]
    },


    "Backend Developer": {
        "Technical": [
            "What is an API?",
            "Explain REST API.",
            "What is the difference between GET and POST?",
            "What is authentication?",
            "What is authorization?",
            "What is a database transaction?",
            "Explain SQL joins.",
            "What is indexing in a database?",
            "What is caching?",
            "How would you design a backend system for a large application?",
            "How would you secure an API?",
            "What is the difference between SQL and NoSQL databases?"
        ],

        "HR": [
            "Tell me about yourself.",
            "Why do you want to work as a backend developer?",
            "What backend technology are you currently learning?",
            "Tell me about a backend project you have worked on."
        ],

        "Behavioral": [
            "Tell me about a backend problem that took time to solve.",
            "Describe a situation where you had to debug a difficult issue.",
            "Tell me about a time you worked under a tight deadline."
        ]
    },


    "Full Stack Developer": {
        "Technical": [
            "Explain the basic architecture of a full stack web application.",
            "How does the frontend communicate with the backend?",
            "What is REST API?",
            "What is authentication and authorization?",
            "Explain SQL joins.",
            "What is database normalization?",
            "What is the difference between frontend and backend validation?",
            "How would you secure a full stack application?",
            "How would you deploy a full stack application?",
            "How would you improve the performance of a full stack application?"
        ],

        "HR": [
            "Tell me about yourself.",
            "Why do you want to become a full stack developer?",
            "Tell me about a full stack project you have built.",
            "What technology would you like to learn next?"
        ],

        "Behavioral": [
            "Tell me about a difficult problem you solved in a web project.",
            "Describe a time when you had to learn a new technology quickly.",
            "Tell me about a disagreement you had while working on a project."
        ]
    },


    "Data Analyst": {
        "Technical": [
            "What is the difference between mean, median and mode?",
            "What is SQL and why is it important for data analysts?",
            "Explain INNER JOIN and LEFT JOIN.",
            "What is data cleaning?",
            "What is an outlier?",
            "What is correlation?",
            "What is the difference between correlation and causation?",
            "How is Python used in data analysis?",
            "What is Pandas?",
            "What is data visualization?",
            "How would you handle missing values in a dataset?"
        ],

        "HR": [
            "Tell me about yourself.",
            "Why do you want to become a data analyst?",
            "Why are you interested in working with data?",
            "What data analysis tools are you learning?"
        ],

        "Behavioral": [
            "Tell me about a time you solved a problem using data.",
            "Describe a situation where your analysis gave an unexpected result.",
            "Tell me about a time you had to explain technical information to a non-technical person."
        ]
    },


    "Machine Learning Engineer": {
        "Technical": [
            "What is Machine Learning?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain overfitting and underfitting.",
            "What is train-test split?",
            "What is cross-validation?",
            "What is linear regression?",
            "What is logistic regression?",
            "What is a decision tree?",
            "What is a confusion matrix?",
            "Explain precision, recall and F1-score.",
            "What is feature engineering?",
            "How would you handle missing data in a machine learning dataset?"
        ],

        "HR": [
            "Tell me about yourself.",
            "Why are you interested in Machine Learning?",
            "What Machine Learning project have you worked on?",
            "Which ML technology are you currently learning?"
        ],

        "Behavioral": [
            "Tell me about a difficult Machine Learning problem you faced.",
            "Describe a time when your model did not perform as expected.",
            "Tell me about a time you had to learn a difficult technical concept."
        ]
    }
}

def evaluate_interview(answers):

    if not answers:
        return {
            "score": 0,
            "average_words": 0,
            "strength": "No answers submitted.",
            "improvement": "Try answering each question clearly."
        }

    total_words = 0
    scores = []

    for item in answers:

        answer = item.get("answer", "").strip()

        words = len(answer.split())

        total_words += words

        # Rule-based practice scoring
        if words < 20:
            score = 35

        elif words < 40:
            score = 55

        elif words < 70:
            score = 70

        elif words < 120:
            score = 82

        else:
            score = 90

        scores.append(score)

    average_score = round(
        sum(scores) / len(scores)
    )

    average_words = round(
        total_words / len(answers)
    )

    if average_words >= 70:

        strength = (
            "Your answers are detailed and "
            "well explained."
        )

    elif average_words >= 40:

        strength = (
            "Your answers provide a reasonable "
            "level of explanation."
        )

    else:

        strength = (
            "You answered the questions, but "
            "your responses could be more detailed."
        )

    if average_words < 40:

        improvement = (
            "Try giving more detailed answers "
            "with examples, reasoning and "
            "relevant technical concepts."
        )

    else:

        improvement = (
            "Focus on making your answers more "
            "structured, precise and interview-oriented."
        )

    return {
        "score": average_score,
        "average_words": average_words,
        "strength": strength,
        "improvement": improvement
    }


@app.route("/mock-interview", methods=["GET", "POST"])
def mock_interview():

    if "user_id" not in session:
        return redirect(url_for("login"))

    target_role = session.get("interview_role")
    interview_type = session.get("interview_type")
    difficulty = session.get("interview_difficulty")

    if not target_role or not interview_type or not difficulty:
        return redirect(url_for("interview_prep"))

    # ---------------- START NEW INTERVIEW ----------------

    if request.method == "GET":

        if interview_type == "Mixed":

            questions = []

            questions.extend(
                INTERVIEW_QUESTIONS
                .get(target_role, {})
                .get("Technical", [])
            )

            questions.extend(
                INTERVIEW_QUESTIONS
                .get(target_role, {})
                .get("HR", [])
            )

            questions.extend(
                INTERVIEW_QUESTIONS
                .get(target_role, {})
                .get("Behavioral", [])
            )

        else:

            questions = (
                INTERVIEW_QUESTIONS
                .get(target_role, {})
                .get(interview_type, [])
            )

        if not questions:
            return "No interview questions available for this selection."

        import random
        import time

        questions = questions.copy()
        random.shuffle(questions)
        # Keep the mock interview limited to 10 questions
        questions = questions[:10]

        session["interview_questions"] = questions
        session["current_question"] = 0
        session["interview_answers"] = []

        # Start one timer for the complete interview
        session["interview_start_time"] = time.time()

    # ---------------- GET SAVED INTERVIEW ----------------

    questions = session.get(
        "interview_questions",
        []
    )

    current_question = session.get(
        "current_question",
        0
    )

    answers = session.get(
        "interview_answers",
        []
    )

    if not questions:
        return redirect(url_for("interview_prep"))
        # Prevent invalid question index

    if current_question >= len(questions):

        evaluation = evaluate_interview(
            answers
        )

        session.pop(
            "interview_start_time",
            None
        )

        return render_template(
            "interview_complete.html",
            target_role=target_role,
            interview_type=interview_type,
            difficulty=difficulty,
            answers=answers,
            total_questions=len(questions),
            evaluation=evaluation
        )

    # ---------------- CALCULATE REMAINING TIME ----------------

    import time

    start_time = session.get(
        "interview_start_time",
        time.time()
    )

    total_interview_time = 20 * 60

    elapsed_time = int(
        time.time() - start_time
    )

    remaining_seconds = max(
        0,
        total_interview_time - elapsed_time
    )

        # ---------------- SUBMIT ANSWER ----------------

    if request.method == "POST":

        answer = request.form.get(
            "answer",
            ""
        ).strip()

        if not answer:

            return render_template(
                "mock_interview.html",
                target_role=target_role,
                interview_type=interview_type,
                difficulty=difficulty,
                question=questions[current_question],
                question_number=current_question + 1,
                total_questions=len(questions),
                remaining_seconds=remaining_seconds,
                error="Please enter your answer before continuing."
            )

        # Save answer

        answers.append({
            "question": questions[current_question],
            "answer": answer
        })

        session["interview_answers"] = answers

        # Move to next question

        current_question += 1

        session["current_question"] = current_question


        # ---------------- INTERVIEW COMPLETE ----------------

        if current_question >= len(questions):

            evaluation = evaluate_interview(
                answers
            )

            session.pop(
                "interview_start_time",
                None
            )

            return render_template(
                "interview_complete.html",
                target_role=target_role,
                interview_type=interview_type,
                difficulty=difficulty,
                answers=answers,
                total_questions=len(questions),
                evaluation=evaluation
            )


    # ---------------- SHOW CURRENT QUESTION ----------------

    return render_template(
        "mock_interview.html",
        target_role=target_role,
        interview_type=interview_type,
        difficulty=difficulty,
        question=questions[current_question],
        question_number=current_question + 1,
        total_questions=len(questions),
        remaining_seconds=remaining_seconds
    )

    # ---------------- EXIT INTERVIEW ----------------

@app.route("/exit-interview")
def exit_interview():

    if "user_id" not in session:
        return redirect(url_for("login"))

    session.pop("interview_questions", None)
    session.pop("current_question", None)
    session.pop("interview_answers", None)
    session.pop("interview_start_time", None)

    return redirect(url_for("interview_prep"))


   


# ---------------- INTERVIEW PREPARATION ----------------
@app.route("/interview-prep", methods=["GET", "POST"])
def interview_prep():

    if "user_id" not in session:
        return redirect(url_for("login"))

    result = None
    message = None

    roles = [
        "Software Developer",
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Developer",
        "Data Analyst",
        "Machine Learning Engineer"
    ]

    interview_types = [
        "Technical",
        "HR",
        "Behavioral",
        "Mixed"
    ]

    difficulties = [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]

    if request.method == "POST":

        target_role = request.form.get(
            "target_role",
            ""
        ).strip()

        interview_type = request.form.get(
            "interview_type",
            ""
        ).strip()

        difficulty = request.form.get(
            "difficulty",
            ""
        ).strip()

        if not target_role or not interview_type or not difficulty:

            message = (
                "Please select all interview preferences."
            )

        else:

            session["interview_role"] = target_role

            session["interview_type"] = interview_type

            session["interview_difficulty"] = difficulty

            return redirect(
                url_for("mock_interview")
            )

    return render_template(
        "interview_prep.html",
        result=result,
        message=message,
        roles=roles,
        interview_types=interview_types,
        difficulties=difficulties
    )

   
# ---------------- CAREER PROFILE ----------------

@app.route("/profile", methods=["GET", "POST"])
def profile():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    # ==============================
    # UPDATE PROFILE
    # ==============================

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()

        if not name or not email:
            conn.close()
            return "Name and email are required."

        try:

            conn.execute(
                """
                UPDATE users
                SET name = ?, email = ?
                WHERE id = ?
                """,
                (name, email, session["user_id"])
            )

            conn.commit()

            # Update current session name
            session["name"] = name

        except Exception as e:

            conn.close()

            return "Could not update profile: " + str(e)


    # ==============================
    # GET USER DATA
    # ==============================

    user = conn.execute(
        """
        SELECT id, name, email
        FROM users
        WHERE id = ?
        """,
        (session["user_id"],)
    ).fetchone()


    # ==============================
    # RESUME DATA
    # ==============================

    resume = conn.execute(
        """
        SELECT file_path
        FROM resumes
        WHERE user_id = ?
        ORDER BY uploaded_at DESC
        LIMIT 1
        """,
        (session["user_id"],)
    ).fetchone()


    # ==============================
    # JOB MATCH DATA
    # ==============================

    match_data = conn.execute(
        """
        SELECT AVG(match_score) AS avg_match
        FROM job_matches
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()


    # ==============================
    # APPLICATION DATA
    # ==============================

    applications_count = conn.execute(
        """
        SELECT COUNT(*) AS total
        FROM applications
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()


    conn.close()


    # ==============================
    # DEFAULT VALUES
    # ==============================

    if not user:
        return redirect(url_for("logout"))


    resume_score = 0
    skills_count = 0
    resume_uploaded = False


    # ==============================
    # ANALYZE UPLOADED RESUME
    # ==============================

    if resume:

        resume_uploaded = True

        if os.path.exists(resume["file_path"]):

            try:

                resume_text = extract_resume_text(
                    resume["file_path"]
                )

                if resume_text.strip():

                    analysis = analyze_resume(resume_text)

                    resume_score = analysis.get(
                        "score",
                        0
                    )

                    skills = extract_skills(
                        resume_text.lower()
                    )

                    skills_count = len(skills)

            except Exception:

                resume_score = 0
                skills_count = 0


    # ==============================
    # AVERAGE JOB MATCH
    # ==============================

    avg_match = 0

    if match_data and match_data["avg_match"] is not None:

        avg_match = round(
            match_data["avg_match"]
        )


    # ==============================
    # RENDER PROFILE
    # ==============================

    return render_template(
        "profile.html",

        user=user,

        resume_score=resume_score,

        skills_count=skills_count,

        avg_match=avg_match,

        applications_count=applications_count["total"],

        resume_uploaded=resume_uploaded
    )
# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))

# ---------------- START APP ----------------

if __name__ == "__main__":

    init_db()

    app.run(debug=True)

# VisionGuard — End-to-End MLOps Pipeline

[![VisionGuard CI](https://github.com/nihaal237/23I-0831_A2_Ops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nihaal237/23I-0831_A2_Ops/actions/workflows/ci.yml)

**AI-Generated vs Real Image Classification | MLOps Assignment 2**

VisionGuard is an end-to-end machine learning application that classifies uploaded images as **AI-generated** or **real**.

The project covers the machine learning lifecycle: dataset preparation, model training, evaluation, experiment tracking, data and model versioning, Flask application development, database integration, containerization, continuous integration, and public cloud deployment.

The application supports two inference modes:

- **Local inference:** Run the TensorFlow model directly inside the local Flask application.
- **Cloud inference:** Send images from local Flask to a publicly deployed Vercel API, which forwards inference requests to a model hosted on Hugging Face Spaces.

Prediction history is stored in a database and displayed through a web interface.

---

## 1. Project Links

| Resource | Link |
|---|---|
| GitHub Repository | [VisionGuard GitHub](https://github.com/nihaal237/23I-0831_A2_Ops) |
| Live Application | [VisionGuard on Vercel](https://23-i-0831-a2-ops.vercel.app/) |
| Hosted Model | [Hugging Face Spaces](https://huggingface.co/spaces/nihaal237/23I-0831-A2-Ops) |
| DagsHub Repository | [VisionGuard DagsHub](https://dagshub.com/nihaal237/23I-0831_A2_Ops) |
| CI Workflow | [GitHub Actions](https://github.com/nihaal237/23I-0831_A2_Ops/actions/workflows/ci.yml) |
| CI Test Project | [Toy CI Repository](https://github.com/nihaal237/23I-0831_A2_Ops_CI_Toy) |

**Cloud deployment approach:** Pathway B — Vercel frontend/API with external model inference and managed PostgreSQL.

---

## 2. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| Model Architecture | MobileNetV2 |
| Image Processing | Pillow, NumPy |
| Web Application | Flask |
| Local Database | SQLite |
| Containerized Database | PostgreSQL 16 |
| Cloud Database | Neon PostgreSQL |
| Database ORM | Flask-SQLAlchemy |
| Experiment Tracking | MLflow / DagsHub |
| Data and Model Versioning | DVC |
| Containerization | Docker / Docker Compose |
| Automation | GitHub Actions |
| Code Quality | Ruff |
| Cloud Frontend | Vercel |
| Hosted Model Inference | Hugging Face Spaces / Gradio |
| API Testing | Postman |
| Version Control | Git / GitHub |

---

## 3. Main Features

- Binary image classification: AI-generated or real.
- Transfer learning and fine-tuning using MobileNetV2.
- Image preprocessing and prediction confidence reporting.
- Model experiment tracking.
- Data and model version control.
- Web-based image upload interface.
- Prediction history with database persistence.
- Image validation and upload size limits.
- Local and cloud inference modes.
- Dockerized Flask and PostgreSQL services.
- CI automation for linting, testing, and Docker builds.
- Public REST API deployment.
- Cloud PostgreSQL integration.
- Postman verification of public API endpoints.

---

## 4. System Architecture

### 4.1 Local Inference

```text
User
  |
  v
Flask Web Interface
  |
  v
Image Validation
  |
  v
TensorFlow / MobileNetV2
  |
  v
Prediction Result
  |
  v
SQLite Database
  |
  v
Prediction History
```

### 4.2 Docker Deployment

```text
Docker Compose
  |
  +---- Flask Container
  |        |
  |        +---- TensorFlow Model
  |        |
  |        +---- Prediction API
  |
  +---- PostgreSQL Container
           |
           +---- Persistent Named Volume
```

### 4.3 Public Cloud Deployment

```text
Browser / Postman
       |
       v
Vercel Flask Application
       |
       +----------------------+
       |                      |
       v                      v
Hugging Face Spaces      Neon PostgreSQL
       |                      |
       v                      v
MobileNetV2 Inference    Prediction History
       |
       v
Prediction Response
```

### 4.4 Local Flask to Cloud Integration

```text
User Upload
    |
    v
Local Flask Application
http://127.0.0.1:5001
    |
    | HTTPS POST /predict
    v
Vercel Cloud API
    |
    v
Hugging Face Model
    |
    v
Prediction Response
    |
    v
Local Flask Application
    |
    +---- Display Prediction
    |
    +---- Store Local History

Vercel also stores successful API predictions
in Neon PostgreSQL.
```

The default local application port is `5000`. Port `5001` may be used if another service is occupying `5000`.

---

## 5. Installation and Setup

### 5.1 Prerequisites

Install:

- Python 3.11
- Git
- pip
- A web browser

Optional, depending on the workflow:

- Docker Desktop for containerization.
- DVC for downloading tracked model artifacts.
- Postman for API testing.
- A DagsHub account for remote artifact access.

### 5.2 Clone the Repository

```powershell
git clone https://github.com/nihaal237/23I-0831_A2_Ops.git
cd 23I-0831_A2_Ops
```

### 5.3 Create a Virtual Environment

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 5.4 Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For DVC operations, install DVC if it is not already available:

```powershell
pip install dvc
```

The full requirements file contains the machine learning, Flask, and supporting dependencies needed for local development.

### 5.5 Configure Environment Variables

Create a local `.env` file from the provided example.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

An example configuration is:

```dotenv
MODEL_PATH=models/mobilenetv2_tuned.keras
PREDICTION_THRESHOLD=0.70
MAX_UPLOAD_MB=10
FLASK_DEBUG=false

DATABASE_URL=sqlite:///visionguard.db

POSTGRES_PASSWORD=ReplaceWithAStrongLocalPassword

INFERENCE_MODE=local
CLOUD_API_URL=https://23-i-0831-a2-ops.vercel.app/predict
```

**Security:** Never commit `.env`, database passwords, DagsHub tokens, or other secrets to GitHub.

---

## 6. Run the Project Locally

### Option A — Cloud Inference (Recommended for Task 8)

This mode uses the publicly hosted model and does not require downloading the TensorFlow checkpoint for inference.

Edit your root `.env`:

```dotenv
INFERENCE_MODE=cloud
DATABASE_URL=sqlite:///visionguard.db
CLOUD_API_URL=https://23-i-0831-a2-ops.vercel.app/predict
```

Start Flask:

```powershell
python -m app.main
```

Open:

http://127.0.0.1:5000/

Health check:

http://127.0.0.1:5000/health

Prediction history:

http://127.0.0.1:5000/history

#### If port 5000 is occupied

On Windows, another service such as WSL or Docker may already be using port 5000.

Stop the existing Flask process and run the application on port 5001:

```powershell
python -c "from app.main import app; app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)"
```

Then open:

http://127.0.0.1:5001/

Health:

http://127.0.0.1:5001/health

History:

http://127.0.0.1:5001/history

This is also the configuration used to demonstrate Task 8.

### Option B — Local TensorFlow Inference

This mode runs the trained MobileNetV2 checkpoint on your computer.

Ensure the model is available:

```text
models/mobilenetv2_tuned.keras
```

If the checkpoint is missing, use DVC to retrieve tracked model artifacts:

```powershell
dvc pull
```

Remote DVC access must already be configured, and private remotes may require authentication.

Set the root `.env`:

```dotenv
INFERENCE_MODE=local
MODEL_PATH=models/mobilenetv2_tuned.keras
PREDICTION_THRESHOLD=0.70
DATABASE_URL=sqlite:///visionguard.db
```

Start the application:

```powershell
python -m app.main
```

Open:

http://127.0.0.1:5000/

The application loads the local model at startup and performs predictions without calling Vercel.

---

## 7. Run Using Docker Compose

Docker Compose runs the Flask application and PostgreSQL in separate containers.

### Prerequisites

- Docker Desktop installed and running.
- Docker Compose available.
- Trained model checkpoint available locally.
- `POSTGRES_PASSWORD` configured in the root `.env`.

The Compose deployment uses the local model rather than the external Hugging Face service.

### 7.1 Set the Database Password

In `.env`:

```dotenv
POSTGRES_PASSWORD=ReplaceWithAStrongLocalPassword
```

### 7.2 Retrieve the Model if Necessary

```powershell
dvc pull
```

Verify that this file exists:

```text
models/mobilenetv2_tuned.keras
```

### 7.3 Start the Containers

From the repository root:

```powershell
docker compose up --build -d
```

### 7.4 Check Container Status

```powershell
docker compose ps
```

### 7.5 View Logs

```powershell
docker compose logs -f web
```

To inspect database logs:

```powershell
docker compose logs -f db
```

### 7.6 Open the Application

http://127.0.0.1:5000/

Health:

http://127.0.0.1:5000/health

History:

http://127.0.0.1:5000/history

### 7.7 Stop the Containers

```powershell
docker compose down
```

This preserves the PostgreSQL named volume.

**Warning:** The following command deletes the Compose-managed database volume and its saved records:

```powershell
docker compose down -v
```

Use it only when intentionally resetting the database.

---

## 8. Model and Data Versioning with DVC

DVC is used to track machine learning artifacts separately from Git.

The repository contains:

```text
.dvc/
models.dvc
```

### Check DVC Configuration

```powershell
dvc remote list
```

### Download Tracked Artifacts

```powershell
dvc pull
```

### Check Tracking Status

```powershell
dvc status
```

### Upload Changed Artifacts

After intentionally updating tracked data or model files:

```powershell
dvc add models
dvc push
```

Commit the updated DVC metadata through Git:

```powershell
git add models.dvc
git commit -m "chore: update DVC-tracked model artifacts"
git push origin main
```

Do not commit large model binaries directly unless explicitly required.

Private DVC remotes may require authenticated access.

---

## 9. Experiment Tracking with MLflow and DagsHub

VisionGuard uses MLflow-compatible experiment tracking as part of its model development workflow.

Experiment tracking supports recording:

- Training parameters.
- Evaluation metrics.
- Training and validation results.
- Model versions.
- Experiment artifacts.

DagsHub project:

https://dagshub.com/nihaal237/23I-0831_A2_Ops

To review existing experiments, open the DagsHub repository and navigate to its experiment/MLflow tracking interface.

If rerunning experiment scripts, use the corresponding scripts under the project source directory and configure the required tracking credentials in your local environment.

Never hardcode tokens or passwords into source code.

---

## 10. REST API Documentation

### 10.1 Public Base URL

```text
https://23-i-0831-a2-ops.vercel.app
```

### 10.2 GET /

Opens the public image classification interface.

```http
GET /
```

### 10.3 GET /health

Returns the cloud service health status.

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "VisionGuard Vercel Frontend",
  "inference_provider": "Hugging Face"
}
```

### 10.4 POST /predict

Accepts an image and returns the predicted class and AI probability.

```http
POST /predict
Content-Type: multipart/form-data
```

Required field:

| Field | Type | Description |
|---|---|---|
| `image` | File | JPG, JPEG, or PNG image |

Example successful response:

```json
{
  "success": true,
  "prediction_id": 4,
  "prediction": "Real",
  "ai_probability": 0.0082,
  "threshold": 0.7,
  "model": "MobileNetV2 Tuned"
}
```

The values above are from an example successful test. Actual predictions and IDs depend on the image and database state.

### 10.5 GET /history

Opens the page showing cloud prediction history stored in Neon PostgreSQL.

```http
GET /history
```

### Upload Restrictions

- Supported file types: JPG, JPEG, PNG.
- Cloud upload size limit: approximately 4 MB.
- Local Flask default upload size limit: 10 MB.
- A valid image is required for inference.

For cloud inference, use images smaller than 4 MB even when uploading through the local application.

### Example Request Using cURL

Windows PowerShell:

```powershell
curl.exe -X POST `
  -F "image=@C:\path\to\image.jpg" `
  https://23-i-0831-a2-ops.vercel.app/predict
```

Linux/macOS:

```bash
curl -X POST \
  -F "image=@/path/to/image.jpg" \
  https://23-i-0831-a2-ops.vercel.app/predict
```

Expected HTTP status for a successful request: `200 OK`.

### Common HTTP Status Codes

| Code | Meaning |
|---|---|
| 200 | Successful inference |
| 400 | Missing, empty, or unsupported upload |
| 413 | Upload exceeds allowed size |
| 502 | Cloud inference temporarily unavailable |

---

## 11. Test the API Using Postman

### Prediction Test

1. Open Postman.
2. Create a new HTTP request.
3. Set the method to `POST`.
4. Enter:

```text
https://23-i-0831-a2-ops.vercel.app/predict
```

5. Select **Body > form-data**.
6. Create a field named `image`.
7. Change the field type from `Text` to `File`.
8. Select a JPG or PNG image.
9. Click **Send**.

Expected result:

- HTTP `200 OK`.
- JSON response containing the prediction.
- AI probability and model name.
- Database prediction ID.

### Health Check Test

Create another request:

```http
GET https://23-i-0831-a2-ops.vercel.app/health
```

Expected result:

- HTTP `200 OK`.
- JSON with `"status": "healthy"`.

### Verify Cloud Database History

After submitting an image using Postman, open:

https://23-i-0831-a2-ops.vercel.app/history

The new prediction should appear in the history page.

---

## 12. Cloud Deployment

### 12.1 Hugging Face Spaces

The trained model is hosted on Hugging Face Spaces using a Gradio application.

Space:

https://huggingface.co/spaces/nihaal237/23I-0831-A2-Ops

The model service accepts image inputs and returns:

- Predicted class.
- AI probability.
- Model metadata.

The Space uses TensorFlow inference, allowing the Vercel frontend to avoid bundling a full TensorFlow model.

### 12.2 Vercel

The public application is deployed through Vercel with the project root set to:

```text
vercel_frontend/
```

The Vercel application provides:

- Public web interface.
- Prediction API.
- Health endpoint.
- Prediction history.
- Integration with the external Hugging Face inference service.
- Integration with Neon PostgreSQL.

Required deployment environment variables:

| Variable | Purpose |
|---|---|
| `HF_SPACE_ID` | Hugging Face Space identifier |
| `DATABASE_URL` | Neon PostgreSQL connection URL |

Example:

```dotenv
HF_SPACE_ID=nihaal237/23I-0831-A2-Ops
DATABASE_URL=<YOUR_NEON_POSTGRESQL_CONNECTION_STRING>
```

Store `DATABASE_URL` only in Vercel's secure environment variable settings. Do not commit real credentials to the repository.

### 12.3 Neon PostgreSQL

Neon is used as the managed PostgreSQL database for cloud predictions.

The cloud application stores:

- Uploaded filename.
- Predicted class.
- AI probability.
- Threshold.
- Model name.
- Timestamp.

The database connection is configured through `DATABASE_URL`.

The cloud history endpoint reads previously stored records directly from Neon.

### 12.4 Cloud Database Connection Test

A connection test is included at:

```text
vercel_frontend/test_neon.py
```

Configure `vercel_frontend/.env` with a valid Neon `DATABASE_URL`.

Run from the repository root:

```powershell
python vercel_frontend/test_neon.py
```

Expected behavior:

```text
Connected to Neon PostgreSQL!
Database: neondb
Connection test: 1
```

This requires a valid database connection and installed `psycopg`/`python-dotenv` dependencies.

---

## 13. Testing

VisionGuard includes Flask route tests and database integration tests.

### Run All Tests

From the repository root:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The test suite includes checks for:

- Homepage response.
- Health endpoint.
- Missing image handling.
- Empty image handling.
- Invalid file extensions.
- Invalid image content.
- Oversized uploads.
- Database table creation.
- Prediction persistence.
- Invalid upload behavior.
- Prediction history.

The current suite contains **12 automated tests**.

### Run Ruff Lint Checks

Install Ruff:

```powershell
pip install ruff
```

Run the same selected rules used by CI:

```powershell
ruff check app tests --select E4,E9,F63,F7,F82
```

### Check Python Syntax

```powershell
python -m compileall -q app tests
```

### Build Docker Image

```powershell
docker build -t visionguard-flask:ci .
```

### CI-Specific Model Assets

GitHub Actions runs:

```powershell
python -m tests.prepare_ci_assets
```

This creates temporary assets for CI tests. It does not download the production checkpoint or perform full model training.

---

## 14. Continuous Integration with GitHub Actions

Workflow file:

```text
.github/workflows/ci.yml
```

The workflow is named **VisionGuard CI**.

It runs on:

- Pushes to `main`.
- Pull requests targeting `main`.
- Manual workflow dispatch.

Pipeline stages:

```text
GitHub Push / Pull Request
          |
          v
Checkout Repository
          |
          v
Set Up Python 3.11
          |
          v
Install Dependencies
          |
          v
Generate CI Test Assets
          |
          v
Check Python Syntax
          |
          v
Run Ruff Linter
          |
          v
Run Flask + Database Tests
          |
          v
Build Docker Image
          |
          v
CI Result
```

The pipeline is designed to verify application quality and buildability.

A successful CI workflow does not automatically deploy the application to Vercel.

### View CI Results

https://github.com/nihaal237/23I-0831_A2_Ops/actions

### Secrets Management

Do not commit:

- `.env` files.
- Neon database passwords.
- DagsHub tokens.
- Private API keys.
- Production credentials.

Use GitHub repository secrets and cloud-provider environment settings when credentials are required.

---

## 15. Security and Limitations

VisionGuard is an academic MLOps demonstration.

Security measures include:

- Upload file extension validation.
- Image content validation in the original Flask application.
- Upload size limits.
- Server-side database access.
- Environment-based configuration.
- Exclusion of secrets from Git.
- Non-root Docker application user.
- Automated testing.

Limitations:

- Image classification is probabilistic and may make incorrect predictions.
- The model is not a definitive detector of all AI-generated images.
- Hugging Face inference availability affects cloud predictions.
- Cloud request body limits are smaller than the default local upload limit.
- Public API endpoints should use stronger authentication, rate limiting, and monitoring before production deployment.
- The application is intended for demonstration and educational evaluation.

---

## 16. Conclusion

VisionGuard demonstrates an end-to-end MLOps workflow for classifying AI-generated and real images.

The project combines machine learning development with reproducibility, model serving, persistent storage, containerization, automated testing, and public cloud deployment.

The final system demonstrates both local TensorFlow inference and a distributed cloud architecture involving Flask, Vercel, Hugging Face Spaces, and Neon PostgreSQL.

**Project:** VisionGuard  
**Course:** MLOps — Assignment 2  
**Repository:** https://github.com/nihaal237/23I-0831_A2_Ops  
**Live Application:** https://23-i-0831-a2-ops.vercel.app/

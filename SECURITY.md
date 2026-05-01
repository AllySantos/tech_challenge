# Security Policy

## Supported Versions

Please ensure you are using the latest version of the application. Security updates are currently provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| >= 1.0.x| :white_check_mark: |
| < 1.0.x | :x:                |

## Reporting a Vulnerability

We take the security of our application and users seriously. If you discover a security vulnerability, please **do not open a public issue.** 

Instead, please report it privately by emailing: **[Your Email Address]**
*(Alternatively, you can use GitHub's "Report a vulnerability" feature under the Security tab).*

Please include the following information in your report:
*   A description of the vulnerability.
*   Steps to reproduce the issue.
*   The potential impact of the vulnerability.
*   Any suggested mitigation or fix.

We will acknowledge receipt of your vulnerability report within [e.g., 48 hours] and strive to send you regular updates about our progress.

---

## Project-Specific Security Guidelines

Because this project serves Machine Learning models via a FastAPI backend, please pay special attention to the following areas when contributing:

### 1. API Security (FastAPI)
*   **Authentication & Authorization:** Ensure endpoints serving sensitive predictions or data are protected (e.g., using OAuth2/JWT).
*   **Input Validation:** Leverage Pydantic models to strictly validate all incoming data shapes, types, and bounds before passing them to the ML model. 
*   **Rate Limiting:** Be aware that ML inference can be computationally expensive. API endpoints should be protected against Denial of Service (DoS) attacks.

### 2. Machine Learning Security
*   **Model Deserialization:** Avoid loading model weights using Python's native `pickle` module from untrusted sources, as this can lead to arbitrary code execution. Prefer safer formats like `safetensors` or ONNX where possible.
*   **Data Privacy:** Ensure no personally identifiable information (PII) is inadvertently logged during inference or stored in the `data/raw/` directory if committed to the repository.

### 3. Dependencies
*   Keep core dependencies (`fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `torch`, etc.) up to date to avoid known CVEs.

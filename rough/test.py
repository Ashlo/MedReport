from transformers import pipeline
summarizer = pipeline("summarization", model="Falconsai/medical_summarization", token="")

print(summarizer)

MEDICAL_DOCUMENT = """
Patient Name: John Doe
DOB: 01/01/1970
Date of Visit: 03/14/2024
Consulting Physician: Dr. Jane Smith

Medical History:

Hypertension
Type 2 Diabetes Mellitus
Current Medications:

Lisinopril 10 mg daily
Metformin 500 mg twice a day
Presenting Complaint:
Patient presents with a two-day history of increasing shortness of breath and swelling in the lower extremities.

Examination Findings:

Blood Pressure: 150/95 mmHg
Heart Rate: 88 bpm
Respiratory Rate: 20 breaths/min
Oxygen Saturation: 92% on room air
Physical Exam: Bilateral pedal edema; no rales or wheezing on lung auscultation
Assessment:
Likely exacerbation of heart failure, considering the patient's history of hypertension and signs of fluid overload.

Plan:

Increase Lisinopril to 20 mg daily
Initiate Furosemide 40 mg daily
Arrange for echocardiogram
Follow-up in one week or sooner if symptoms worsen

"""

print(summarizer(MEDICAL_DOCUMENT, max_length=850, min_length=500, do_sample=False))

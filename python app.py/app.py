from __future__ import annotations

import os
import uuid
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for, session, redirect
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)
METADATA_FILE = REPORT_DIR / "reports_index.json"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "kn": "ಕನ್ನಡ",
    "hi": "हिन्दी",
    "ta": "தமிழ்",
    "ml": "മലയാളം",
    "te": "తెలుగు",
}

TRANSLATIONS = {
    "en": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI Image Analysis",
        "heading": "Vitscan AI Prediction",
        "lead": "Upload a face or skin-related image and get a heuristic prediction based on color, contrast, and detected facial patterns.",
        "note": "This is an educational demo. It is not a medical diagnosis and should not replace lab tests or a doctor.",
        "choose_image": "Choose Image",
        "analyze_image": "Analyze Image",
        "language": "Language",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "uploaded_image": "Uploaded Image",
        "visual_indicators": "Visual Indicators",
        "recommendations": "Recommendations",
        "medicines": "Medicine Recommendations",
        "extracted_metrics": "Extracted Metrics",
        "download_report": "Download Full Report",
        "report_title": "Detailed Report",
        "report_message": "A full text report has been generated for your analysis.",
        "error_no_image": "Please choose an image to analyze.",
        "error_invalid": "Upload a PNG, JPG, JPEG, or WEBP image.",
        "home": "Home",
        "view_reports": "View Reports",
        "view_uploads": "View Uploads",
        "patient_name": "Patient Name",
        "patient_age": "Patient Age",
        "patient_gender": "Patient Gender",
        "patient_notes": "Patient Notes",
        "patient_details": "Patient Details",
        "root_cause": "Root Cause",
        "solutions": "Solutions",
        "login": "Login",
        "logout": "Logout",
        "sign_in": "Sign In",
        "username": "Username",
        "password": "Password",
        "admin_portal": "Admin Portal",
        "admin_dashboard": "Admin Dashboard",
    },
    "es": {
        "title": "Vitscan AI",
        "eyebrow": "Análisis de Imágenes Vitscan AI",
        "heading": "Predicción Vitscan AI",
        "lead": "Sube una imagen de rostro o piel y obtén una predicción heurística basada en color, contraste y patrones faciales detectados.",
        "note": "Esto es una demostración educativa. No es un diagnóstico médico y no debe reemplazar análisis de laboratorio ni a un médico.",
        "choose_image": "Seleccionar Imagen",
        "analyze_image": "Analizar Imagen",
        "language": "Idioma",
        "prediction": "Predicción",
        "confidence": "Confianza",
        "uploaded_image": "Imagen Subida",
        "visual_indicators": "Indicadores Visuales",
        "recommendations": "Recomendaciones",
        "medicines": "Recomendaciones de Medicamentos",
        "extracted_metrics": "Métricas Extraídas",
        "download_report": "Descargar Informe Completo",
        "report_title": "Informe Detallado",
        "report_message": "Se ha generado un informe de texto completo para su análisis.",
        "error_no_image": "Por favor elige una imagen para analizar.",
        "error_invalid": "Sube una imagen PNG, JPG, JPEG o WEBP.",
        "home": "Inicio",
        "view_reports": "Ver Informes",
        "view_uploads": "Ver Subidas",
        "patient_name": "Nombre del Paciente",
        "patient_age": "Edad del Paciente",
        "patient_gender": "Género del Paciente",
        "patient_notes": "Notas del Paciente",
        "patient_details": "Detalles del Paciente",
        "root_cause": "Causa Raíz",
        "solutions": "Soluciones",
        "login": "Iniciar Sesión",
        "logout": "Cerrar Sesión",
        "sign_in": "Acceder",
        "username": "Usuario",
        "password": "Contraseña",
        "admin_portal": "Portal de Administrador",
        "admin_dashboard": "Panel de Admin",
    },
    "fr": {
        "title": "Vitscan AI",
        "eyebrow": "Analyse d'Image Vitscan AI",
        "heading": "Prédiction Vitscan AI",
        "lead": "Téléchargez une image de visage ou de peau et obtenez une prédiction heuristique basée sur la couleur, le contraste et les motifs faciaux détectés.",
        "note": "Ceci est une démo pédagogique. Ce n'est pas un diagnostic médical et cela ne doit pas remplacer des analyses de laboratoire ou un médecin.",
        "choose_image": "Choisir une Image",
        "analyze_image": "Analyser l'Image",
        "language": "Langue",
        "prediction": "Prédiction",
        "confidence": "Confiance",
        "uploaded_image": "Image Téléchargée",
        "visual_indicators": "Indicateurs Visuels",
        "recommendations": "Recommandations",
        "medicines": "Recommandations de Médicaments",
        "extracted_metrics": "Métriques Extraites",
        "download_report": "Télécharger le Rapport Complet",
        "report_title": "Rapport Détaillé",
        "report_message": "Un rapport texte complet a été généré pour votre analyse.",
        "error_no_image": "Veuillez choisir une image à analyser.",
        "error_invalid": "Téléchargez une image PNG, JPG, JPEG ou WEBP.",
        "home": "Accueil",
        "view_reports": "Voir les Rapports",
        "view_uploads": "Voir les Téléchargements",
        "patient_name": "Nom du Patient",
        "patient_age": "Âge du Patient",
        "patient_gender": "Sexe du Patient",
        "patient_notes": "Notes du Patient",
        "patient_details": "Informations du Patient",
        "root_cause": "Cause Racine",
        "solutions": "Solutions",
        "login": "Connexion",
        "logout": "Déconnexion",
        "sign_in": "Se Connecter",
        "username": "Nom d'utilisateur",
        "password": "Mot de passe",
        "admin_portal": "Portail Admin",
        "admin_dashboard": "Tableau de Bord Admin",
    },
    "kn": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI ಇಮೇಜ್ ವಿಶ್ಲೇಷಣೆ",
        "heading": "Vitscan AI ಭವಿಷ್ಯ ವಾಣಿ",
        "lead": "ಮುಖ ಅಥವಾ ಚರ್ಮದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಬಣ್ಣ, ಭಿನ್ನತೆ ಮತ್ತು ಮುಖದ ಮಾದರಿಗಳ ಆಧಾರದ ಮೇಲೆ ಅಂದಾಜು ಪಡೆಯಿರಿ.",
        "note": "ಇದು ಶೈಕ್ಷಣಿಕ ಡೆಮೋ ಆಗಿದೆ. ಇದು ವೈದ್ಯಕೀಯ ನಿರ್ಧಾರವಲ್ಲ ಮತ್ತು ಪ್ರಯೋಗಾಲಯದ ಪರೀಕ್ಷೆಗಳನ್ನು ಅಥವಾ ವೈದ್ಯರನ್ನು ಬದಲಾಯಿಸಬಾರದು.",
        "choose_image": "ಚಿತ್ರ ಆಯ್ಕೆಮಾಡಿ",
        "analyze_image": "ವಿಶ್ಲೇಷಿಸಿ",
        "language": "ಭಾಷೆ",
        "prediction": "ಅಂದಾಜು",
        "confidence": "ಭರವಸೆ",
        "uploaded_image": "ಅಪ್‌ಲೋಡ್ ಮಾಡಲಾದ ಚಿತ್ರ",
        "visual_indicators": "ಕಾಣುವ ಸೂಚನೆಗಳು",
        "recommendations": "ಶಿಫಾರಸುಗಳು",
        "extracted_metrics": "ತೀತ ಕണക്കಿತಗಳು",
        "download_report": "पूर्ण ವರದಿ ಡೌನ್ಲೋಡ್ ಮಾಡಿ",
        "report_title": "ವಿಸ್ತೃತ ವರದಿ",
        "report_message": "ನಿಮ್ಮ ವಿಶ್ಲೇಷಣೆಗೆ ಪಠ್ಯ ವರದಿ ಸೃಷ್ಟಿಸಲಾಗಿದೆ.",
        "error_no_image": "ದಯವಿಟ್ಟು ವಿಶ್ಲೇಷಿಸಲು ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.",
        "error_invalid": "PNG, JPG, JPEG ಅಥವಾ WEBP ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ.",
        "home": "ಮೈನ್",
        "view_reports": "ವರದಿಗಳನ್ನು ನೋಡುವುದು",
        "view_uploads": "ಅಪ್‌ಲೋಡ್‌ಗಳನ್ನು ನೋಡಿ",
        "patient_name": "ರೋಗಿಯ ಹೆಸರು",
        "patient_age": "ರೋಗಿಯ ವಯಸ್ಸು",
        "patient_gender": "ಲಿಂಗ",
        "patient_notes": "ರೋಗಿಯ ಟಿಪ್ಪಣಿಗಳು",
        "patient_details": "ರೋಗಿಯ ವಿವರಗಳು",
        "root_cause": "ಮೂಲ ಕಾರಣ",
        "solutions": " ಪರಿಹಾರಗಳು",
        "login": "ಲಾಗಿನ್",
        "logout": "ಲಾಗೌಟ್",
        "sign_in": "ಸೈನ್ ಇನ್",
        "username": "ಬಳಕೆದಾರ ಹೆಸರು",
        "password": "ಗುಪ್ತಪದ",
        "admin_portal": "ಅಡ್ಮಿನ್ ಪೋರ್‍ಟಲ್",
        "admin_dashboard": "ಅಡ್ಮಿನ್ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
    },
    "hi": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI इमेज विश्लेषण",
        "heading": "Vitscan AI भविष्यवाणी",
        "lead": "चेहरे या त्वचा की छवि अपलोड करें और रंग, कंट्रास्ट और पहचाने गए चेहरे के पैटर्न के आधार पर एक अनुमान प्राप्त करें।",
        "note": "यह एक शैक्षिक डेमो है। यह चिकित्सा निदान नहीं है और इसे लैब परीक्षणों या डॉक्टर की जगह नहीं लेना चाहिए।",
        "choose_image": "चित्र चुनें",
        "analyze_image": "विश्लेषण करें",
        "language": "भाषा",
        "prediction": "पूर्वानुमान",
        "confidence": "विश्वास",
        "uploaded_image": "अपलोड की गई छवि",
        "visual_indicators": "दृश्य संकेतक",
        "recommendations": "सिफ़ारिशें",
        "extracted_metrics": "निकाले गए मीट्रिक",
        "download_report": "पूर्ण रिपोर्ट डाउनलोड करें",
        "report_title": "विस्तृत रिपोर्ट",
        "report_message": "आपके विश्लेषण के लिए पूर्ण पाठ रिपोर्ट उत्पन्न की गई है।",
        "error_no_image": "कृपया विश्लेषण के लिए एक छवि चुनें।",
        "error_invalid": "PNG, JPG, JPEG या WEBP छवि अपलोड करें।",
        "home": "होम",
        "view_reports": "रिपोर्ट देखें",
        "view_uploads": "अपलोड देखें",
        "patient_name": "मरीज का नाम",
        "patient_age": "मरीज की आयु",
        "patient_gender": "लिंग",
        "patient_notes": "मरीज के नोट",
        "patient_details": "रोगी विवरण",
        "root_cause": "मूल कारण",
        "solutions": "समाधान",
        "login": "लॉगिन",
        "logout": "लॉगआउट",
        "sign_in": "साइन इन",
        "username": "उपयोगकर्ता नाम",
        "password": "पासवर्ड",
        "admin_portal": "एडमिन पोर्टल",
        "admin_dashboard": "एडमिन डैशबोर्ड",
    },
    "ta": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI படம் பகுப்பு",
        "heading": "Vitscan AI கணிப்பு",
        "lead": "முகம் அல்லது தோல் தொடர்புடைய படத்தை பதிவேற்றவும் மற்றும் நிறம், வண்ணமியக்கம் மற்றும் கண்டறியப்பட்ட முக மாதிரிகளின் அடிப்படையில் ஒரு ஊகத்தைப் பெறவும்.",
        "note": "இது ஒரு கல்வி காட்சி. இது மருத்துவ நோயறிதல் அல்ல மற்றும் ஆய்வக சோதனைகளை அல்லது டாக்டரை மாற்றக்கூடாது.",
        "choose_image": "படத்தைத் தேர்ந்தெடு",
        "analyze_image": "பகுப்பாய்வு",
        "language": "மொழி",
        "prediction": "முன்னறிவு",
        "confidence": "நம்பிக்கை",
        "uploaded_image": "பதிவேற்றப்பட்ட படம்",
        "visual_indicators": "காட்சி குறியீடுகள்",
        "recommendations": "பரிந்துரைகள்",
        "extracted_metrics": "பெறப்பட்ட அளவுருக்கள்",
        "download_report": "முழு அறிக்கையை பதிவிறக்குக",
        "report_title": "விரிவான அறிக்கை",
        "report_message": "உங்கள் பகுப்பாய்வுக்காக முழு உரை அறிக்கை உருவாக்கப்பட்டது.",
        "error_no_image": "தயவு செய்து பகுப்பாய்விற்கு ஒரு படத்தைத் தேர்ந்தெடுக்கவும்.",
        "error_invalid": "PNG, JPG, JPEG அல்லது WEBP படத்தை பதிவேற்றவும்.",
        "home": "முகப்பு",
        "view_reports": "அறிக்கைகளைக் காட்டு",
        "view_uploads": "பதிவேற்றங்களைப் பார்க்கவும்",
        "patient_name": "நோயாளர் பெயர்",
        "patient_age": "நோயாளர் வயது",
        "patient_gender": "பாலினம்",
        "patient_notes": "நோயாளர் குறிப்புகள்",
        "patient_details": "நோயாளர் விவரங்கள்",
        "root_cause": "மூல காரணம்",
        "solutions": "தீர்வுகள்",
        "login": "உள்நுழைவு",
        "logout": "வெளியேறு",
        "sign_in": "உள்நுழையவும்",
        "username": "பயனர் பெயர்",
        "password": "கடவுச்சொல்",
        "admin_portal": "அட்மின் பயன்பாட்டுக்கை",
        "admin_dashboard": "அட்மின் போர்ட்டல்",
    },
    "ml": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI ഇമേജ് വിശകലനം",
        "heading": "Vitscan AI പ്രവചന",
        "lead": "മുഖം അല്ലെങ്കിൽ ത്വക്കുമായി ബന്ധപ്പെട്ട ചിത്രം അപ്‌ലോഡ് ചെയ്ത് നിറം, വ്യത്യാസം, കണ്ടെത്തിയ മുഖ മാതൃകകൾ എന്നിവയുടെ അടിസ്ഥാനത്തിൽ ഒരു കണക്കുകൂട്ടൽ ലഭിക്കുക.",
        "note": "ഇത് ഒരു വിദ്യാഭ്യാസ ഡെമോ ആണ്. ഇത് മെഡിക്കൽ രോഗനിർണയം അല്ല, ലാബ് ടെസ്റ്റുകൾ അല്ലെങ്കിൽ ഡോക്ടറെ மாற்றല്ല.",
        "choose_image": "ചിത്രം തിരഞ്ഞെടുക്കുക",
        "analyze_image": "വിശകലനം ചെയ്യുക",
        "language": "ഭാഷ",
        "prediction": "കണക്കാക്കൽ",
        "confidence": "വിശ്വാസം",
        "uploaded_image": "അപ്‌ലോഡ് ചെയ്ത ചിത്രം",
        "visual_indicators": "ദൃശ്യ സൂചികകൾ",
        "recommendations": "സൂചനകൾ",
        "extracted_metrics": "എടുക്കപ്പെട്ട മെട്രിക്കുകൾ",
        "download_report": "പൂർണ്ണ റിപ്പോർട്ട് ഡൗൺലോഡ് ചെയ്യുക",
        "report_title": "വിവരമുള്ള റിപ്പോർട്ട്",
        "report_message": "നിങ്ങളുടെ വിശകലനത്തിനായി പൂർണ്ണ പാഠ റിപ്പോർട്ട് സൃഷ്ടിച്ചു.",
        "error_no_image": "ദയവായി വിശകലനത്തിനായി ഒരു ചിത്രം തിരഞ്ഞെടുക്കുക.",
        "error_invalid": "PNG, JPG, JPEG, അല്ലെങ്കിൽ WEBP ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.",
        "home": "ഹോം",
        "view_reports": "റിപ്പോർട്ടുകൾ കാണുക",
        "view_uploads": "അപ്‌ലോഡുകൾ കാണുക",
        "patient_name": "രോഗിയുടെ പേര്",
        "patient_age": "രോഗിയുടെ പ്രായം",
        "patient_gender": "ലിംഗം",
        "patient_notes": "രോഗിയുടെ കുറിപ്പുകൾ",
        "patient_details": "രോഗിയുടെ വിവരങ്ങൾ",
        "root_cause": "പ്രധാന കാരണം",
        "solutions": "പരിഹാരങ്ങൾ",
        "login": "ലോഗിൻ",
        "logout": "ലോഗ്ഔട്ട്",
        "sign_in": "സൈൻ ഇൻ",
        "username": "ഉപയോക്തൃ പേര്",
        "password": "പാസ്‌വേഡ്",
        "admin_portal": "അഡ്‌മിൻ പോർട്ടൽ",
        "admin_dashboard": "അഡ്‌മിൻ ഡാഷ്‌ബോർഡ്",
    },
    "te": {
        "title": "Vitscan AI",
        "eyebrow": "Vitscan AI చిత్ర విశ్లేషణ",
        "heading": "Vitscan AI భవిష్యవాణి",
        "lead": "ముఖం లేదా చర్మ సంబంధిత చిత్రాన్ని అప్‌లోడ్ చేసి, రంగు, వ్యత్యాసం మరియు గుర్తించిన ముఖ నమూనాల ఆధారంగా ఊహలను పొందండి.",
        "note": "ఇది విద్యా ప్రదర్శన. ఇది వైద్య నిర్ధారణ కాదు మరియు ప్రయోగశాల పరీక్షలు లేదా డాక్టర్‌ను బదులుగా చేసుకోవద్దు.",
        "choose_image": "చిత్రం ఎంచుకోండి",
        "analyze_image": "విశ్లేషించు",
        "language": "భాష",
        "prediction": "భవిష్యవాణి",
        "confidence": "నిర్భయం",
        "uploaded_image": "అప్‌లోడ్ చేసిన చిత్రం",
        "visual_indicators": "దృశ్య సూచికలు",
        "recommendations": "సిఫార్సులు",
        "extracted_metrics": "పొందిన మీట్రిక్స్",
        "download_report": "పూర్తి నివేదికను డౌన్‌లోడ్ చేయండి",
        "report_title": "వివర నివేదిక",
        "report_message": "మీ విశ్లేషణ కోసం పూర్తి పాఠ నివేదిక రూపొందించబడింది.",
        "error_no_image": "దయచేసి విశ్లేషణకు ఒక చిత్రాన్ని ఎంచుకోండి.",
        "error_invalid": "PNG, JPG, JPEG లేదా WEBP చిత్రం‌ను అప్‌లోడ్ చేయండి.",
        "home": "హోమ్",
        "view_reports": "రిపోర్టులను వీక్షించండి",
        "view_uploads": "అప్‌లోడ్స్ ని చూడండి",
        "patient_name": "రోగి పేరు",
        "patient_age": "వయస్సు",
        "patient_gender": "లింగం",
        "patient_notes": "రోగి నోట్స్",
        "patient_details": "రోగి వివరాలు",
        "root_cause": "మూల కారణం",
        "solutions": "పరిష్కారాలు",
        "login": "లాగిన్",
        "logout": "లాగౌట్",
        "sign_in": "సైన్ ఇన్",
        "username": "వినియోగదారు పేరు",
        "password": "రహస్యపదం",
        "admin_portal": "అడ్మిన్ పోర్టల్",
        "admin_dashboard": "అడ్మిన్ డ్యాష్‌బోర్డ్",
    },
}

METRIC_LABELS = {
    "en": {
        "brightness": "Brightness",
        "contrast": "Contrast",
        "red_ratio": "Red Ratio",
        "yellow_ratio": "Yellow Ratio",
        "saturation": "Saturation",
        "pallor_index": "Pallor Index",
    },
    "es": {
        "brightness": "Brillo",
        "contrast": "Contraste",
        "red_ratio": "Proporción Roja",
        "yellow_ratio": "Proporción Amarilla",
        "saturation": "Saturación",
        "pallor_index": "Índice de Palidez",
    },
    "fr": {
        "brightness": "Luminosité",
        "contrast": "Contraste",
        "red_ratio": "Rapport Rouge",
        "yellow_ratio": "Rapport Jaune",
        "saturation": "Saturation",
        "pallor_index": "Indice de Pâleur",
    },
    "kn": {},
    "hi": {},
    "ta": {},
    "ml": {},
    "te": {},
}

LABEL_TRANSLATIONS = {
    "en": {
        "Iron deficiency / anemia": "Iron deficiency / anemia",
        "Vitamin B12 deficiency": "Vitamin B12 deficiency",
        "Vitamin A deficiency": "Vitamin A deficiency",
        "Vitamin C deficiency": "Vitamin C deficiency",
        "No strong visual deficiency signal": "No strong visual deficiency signal",
    },
    "es": {
        "Iron deficiency / anemia": "Deficiencia de hierro / anemia",
        "Vitamin B12 deficiency": "Deficiencia de vitamina B12",
        "Vitamin A deficiency": "Deficiencia de vitamina A",
        "Vitamin C deficiency": "Deficiencia de vitamina C",
        "No strong visual deficiency signal": "Sin señal visual fuerte de deficiencia",
    },
    "fr": {
        "Iron deficiency / anemia": "Carence en fer / anémie",
        "Vitamin B12 deficiency": "Carence en vitamine B12",
        "Vitamin A deficiency": "Carence en vitamine A",
        "Vitamin C deficiency": "Carence en vitamine C",
        "No strong visual deficiency signal": "Pas de signal visuel fort de carence",
    },
}

INDICATOR_TRANSLATIONS = {
    "en": {
        "High facial brightness with reduced red tone suggests pallor.": "High facial brightness with reduced red tone suggests pallor.",
        "Central facial area appears lighter than surrounding region.": "Central facial area appears lighter than surrounding region.",
        "Higher red emphasis and contrast may align with inflamed mouth or tongue regions.": "Higher red emphasis and contrast may align with inflamed mouth or tongue regions.",
        "Lower contrast and lower saturation can reflect dull or dry-looking skin.": "Lower contrast and lower saturation can reflect dull or dry-looking skin.",
        "Yellow-red imbalance may indicate gum or skin irritation patterns.": "Yellow-red imbalance may indicate gum or skin irritation patterns.",
        "The analysis did not find a strong visual abnormality in the extracted image region.": "The analysis did not find a strong visual abnormality in the extracted image region.",
    },
    "es": {
        "High facial brightness with reduced red tone suggests pallor.": "El alto brillo facial con tono rojo reducido sugiere palidez.",
        "Central facial area appears lighter than surrounding region.": "La zona central del rostro parece más clara que la región circundante.",
        "Higher red emphasis and contrast may align with inflamed mouth or tongue regions.": "Un mayor énfasis en el rojo y el contraste puede coincidir con regiones inflamadas de la boca o la lengua.",
        "Lower contrast and lower saturation can reflect dull or dry-looking skin.": "El menor contraste y la menor saturación pueden reflejar piel apagada o seca.",
        "Yellow-red imbalance may indicate gum or skin irritation patterns.": "El desequilibrio amarillo-rojo puede indicar patrones de irritación de encías o piel.",
        "The analysis did not find a strong visual abnormality in the extracted image region.": "El análisis no encontró una anomalía visual fuerte en la región de imagen extraída.",
    },
    "fr": {
        "High facial brightness with reduced red tone suggests pallor.": "Une luminosité faciale élevée avec un ton rouge réduit suggère une pâleur.",
        "Central facial area appears lighter than surrounding region.": "La zone centrale du visage semble plus claire que la région environnante.",
        "Higher red emphasis and contrast may align with inflamed mouth or tongue regions.": "Une plus grande priorité au rouge et au contraste peut correspondre à des zones inflammées de la bouche ou de la langue.",
        "Lower contrast and lower saturation can reflect dull or dry-looking skin.": "Un contraste et une saturation plus faibles peuvent refléter une peau terne ou sèche.",
        "Yellow-red imbalance may indicate gum or skin irritation patterns.": "Un déséquilibre jaune-rouge peut indiquer des motifs d'irritation des gencives ou de la peau.",
        "The analysis did not find a strong visual abnormality in the extracted image region.": "L'analyse n'a pas trouvé d'anomalie visuelle forte dans la région d'image extraite.",
    },
}

SUMMARIES = {
    "en": {
        "Iron deficiency / anemia": "The image shows pallor-like patterns that can be associated with low iron or anemia.",
        "Vitamin B12 deficiency": "The image shows red and inflamed color patterns that may align with Vitamin B12-related signs.",
        "Vitamin A deficiency": "The image looks lower in contrast and saturation, which can match dry or dull visual symptoms.",
        "Vitamin C deficiency": "The color balance suggests irritation-related patterns sometimes seen with low Vitamin C.",
        "No strong visual deficiency signal": "The uploaded image does not show a strong match to the built-in visual deficiency heuristics.",
    },
    "es": {
        "Iron deficiency / anemia": "La imagen muestra patrones similares a la palidez que pueden asociarse con bajo hierro o anemia.",
        "Vitamin B12 deficiency": "La imagen muestra patrones rojos e inflamados que pueden coincidir con signos relacionados con la vitamina B12.",
        "Vitamin A deficiency": "La imagen parece tener menor contraste y saturación, lo que puede coincidir con síntomas visuales secos o apagados.",
        "Vitamin C deficiency": "El equilibrio de color sugiere patrones de irritación a veces vistos con bajo contenido de vitamina C.",
        "No strong visual deficiency signal": "La imagen subida no muestra una coincidencia fuerte con las heurísticas visuales de deficiencia.",
    },
    "fr": {
        "Iron deficiency / anemia": "L'image montre des motifs de pâleur qui peuvent être associés à une carence en fer ou à l'anémie.",
        "Vitamin B12 deficiency": "L'image montre des motifs rouges et enflammés qui peuvent correspondre à des signes liés à la vitamine B12.",
        "Vitamin A deficiency": "L'image semble avoir moins de contraste et de saturation, ce qui peut correspondre à des symptômes visuels secs ou ternes.",
        "Vitamin C deficiency": "L'équilibre des couleurs suggère des motifs d'irritation parfois vus en cas de faible teneur en vitamine C.",
        "No strong visual deficiency signal": "L'image téléchargée ne montre pas de correspondance forte avec les heuristiques visuelles de carence.",
    },
}

ADVICE_TEXTS = {
    "en": {
        "Iron deficiency / anemia": [
            "Consider iron-rich foods such as spinach, beans, red meat, and lentils.",
            "Pair plant-based iron sources with Vitamin C-rich foods to improve absorption.",
            "Seek a blood test before taking supplements.",
        ],
        "Vitamin B12 deficiency": [
            "Discuss B12 testing with a clinician, especially if fatigue or numbness is present.",
            "Common sources include eggs, dairy, fish, and fortified cereals.",
            "Avoid self-diagnosing from images alone.",
        ],
        "Vitamin A deficiency": [
            "Include carrots, sweet potatoes, leafy greens, and eggs in your diet.",
            "If night-vision issues or severe dryness are present, seek medical advice.",
            "A photo cannot confirm deficiency on its own.",
        ],
        "Vitamin C deficiency": [
            "Add citrus fruits, berries, tomatoes, and peppers to meals.",
            "Persistent gum bleeding or bruising should be evaluated clinically.",
            "Lab confirmation is recommended before supplementation.",
        ],
        "No strong visual deficiency signal": [
            "The image does not strongly suggest deficiency, but symptoms still matter.",
            "Maintain a balanced diet with fruits, vegetables, protein, and hydration.",
            "Use lab tests for reliable confirmation.",
        ],
    },
    "es": {
        "Iron deficiency / anemia": [
            "Consume alimentos ricos en hierro como espinacas, frijoles, carne roja y lentejas.",
            "Combina fuentes vegetales de hierro con alimentos ricos en vitamina C para mejorar la absorción.",
            "Busca un análisis de sangre antes de tomar suplementos.",
        ],
        "Vitamin B12 deficiency": [
            "Consulta una prueba de B12 con un clínico, especialmente si hay fatiga o entumecimiento.",
            "Fuentes comunes incluyen huevos, lácteos, pescado y cereales fortificados.",
            "Evita el autodiagnóstico solo con imágenes.",
        ],
        "Vitamin A deficiency": [
            "Incluye zanahorias, batatas, verduras de hoja verde y huevos en tu dieta.",
            "Si hay problemas de visión nocturna o sequedad severa, busca consejo médico.",
            "Una foto no puede confirmar una deficiencia por sí sola.",
        ],
        "Vitamin C deficiency": [
            "Agrega cítricos, bayas, tomates y pimientos a las comidas.",
            "El sangrado de encías persistente o moretones debe evaluarse clínicamente.",
            "Se recomienda confirmación de laboratorio antes de suplementar.",
        ],
        "No strong visual deficiency signal": [
            "La imagen no sugiere fuertemente una deficiencia, pero los síntomas siguen siendo importantes.",
            "Mantén una dieta equilibrada con frutas, verduras, proteínas e hidratación.",
            "Usa pruebas de laboratorio para una confirmación confiable.",
        ],
    },
    "fr": {
        "Iron deficiency / anemia": [
            "Considérez des aliments riches en fer comme les épinards, les haricots, la viande rouge et les lentilles.",
            "Associez les sources végétales de fer à des aliments riches en vitamine C pour améliorer l'absorption.",
            "Faites un test sanguin avant de prendre des suppléments.",
        ],
        "Vitamin B12 deficiency": [
            "Discutez d'un test B12 avec un clinicien, surtout en cas de fatigue ou d'engourdissement.",
            "Les sources courantes incluent les œufs, les produits laitiers, le poisson et les céréales enrichies.",
            "Évitez l'auto-diagnostic uniquement à partir d'images.",
        ],
        "Vitamin A deficiency": [
            "Incluez des carottes, des patates douces, des légumes verts à feuilles et des œufs dans votre alimentation.",
            "Si des problèmes de vision nocturne ou une sécheresse sévère sont présents, consultez un médecin.",
            "Une photo ne peut pas confirmer une carence à elle seule.",
        ],
        "Vitamin C deficiency": [
            "Ajoutez des agrumes, des baies, des tomates et des poivrons aux repas.",
            "Un saignement persistant des gencives ou des ecchymoses doit être évalué cliniquement.",
            "Une confirmation en laboratoire est recommandée avant la supplémentation.",
        ],
        "No strong visual deficiency signal": [
            "L'image ne suggère pas fortement une carence, mais les symptômes restent importants.",
            "Maintenez une alimentation équilibrée avec des fruits, des légumes, des protéines et une hydratation.",
            "Utilisez des tests de laboratoire pour une confirmation fiable.",
        ],
    },
}

MEDICINE_TEXTS = {
    "en": {
        "Iron deficiency / anemia": [
            "Discuss iron supplements such as ferrous sulfate with a clinician.",
            "Consider a multivitamin with iron if recommended by a doctor.",
            "Ask your provider about iron formulations that minimize stomach upset.",
        ],
        "Vitamin B12 deficiency": [
            "Discuss cyanocobalamin or methylcobalamin supplementation with a clinician.",
            "B12 injections may be recommended if absorption is impaired.",
            "Check if a complete B-complex is appropriate for your situation.",
        ],
        "Vitamin A deficiency": [
            "Discuss a vitamin A supplement with your healthcare provider.",
            "Ask if a beta-carotene or retinol formulation is best for you.",
            "Avoid high-dose vitamin A without medical supervision.",
        ],
        "Vitamin C deficiency": [
            "Discuss ascorbic acid supplements with a clinician.",
            "Consider a buffered vitamin C formula if you have stomach sensitivity.",
            "Ask your provider whether a daily 500mg to 1000mg dose is appropriate.",
        ],
        "No strong visual deficiency signal": [
            "No specific supplement is recommended based on the image alone.",
            "Focus on a balanced diet and speak with a clinician before supplementing.",
            "Use lab tests to decide whether any supplements are necessary.",
        ],
    },
    "es": {
        "Iron deficiency / anemia": [
            "Consulte suplementos de hierro como sulfato ferroso con un clínico.",
            "Considere un multivitamínico con hierro si lo recomienda un médico.",
            "Pregunte a su proveedor sobre formulaciones de hierro que minimicen molestias estomacales.",
        ],
        "Vitamin B12 deficiency": [
            "Discuta la suplementación con cianocobalamina o metilcobalamina con un clínico.",
            "Las inyecciones de B12 pueden recomendarse si la absorción es deficiente.",
            "Verifique si un complejo B completo es apropiado para su situación.",
        ],
        "Vitamin A deficiency": [
            "Consulte un suplemento de vitamina A con su proveedor de atención médica.",
            "Pregunte si una formulación de betacaroteno o retinol es mejor para usted.",
            "Evite altas dosis de vitamina A sin supervisión médica.",
        ],
        "Vitamin C deficiency": [
            "Discuta suplementos de ácido ascórbico con un clínico.",
            "Considere una fórmula de vitamina C amortiguada si tiene sensibilidad estomacal.",
            "Pregunte a su proveedor si una dosis diaria de 500 mg a 1000 mg es apropiada.",
        ],
        "No strong visual deficiency signal": [
            "No se recomienda ningún suplemento específico solo con la imagen.",
            "Concéntrese en una dieta equilibrada y hable con un clínico antes de suplementar.",
            "Use pruebas de laboratorio para decidir si necesita suplementos.",
        ],
    },
    "fr": {
        "Iron deficiency / anemia": [
            "Discutez des suppléments de fer comme le sulfate ferreux avec un clinicien.",
            "Envisagez un multivitamine avec du fer si un médecin le recommande.",
            "Demandez à votre prestataire des formulations de fer qui minimisent les maux d'estomac.",
        ],
        "Vitamin B12 deficiency": [
            "Discutez de la supplémentation en cyanocobalamine ou méthylcobalamine avec un clinicien.",
            "Les injections de B12 peuvent être recommandées en cas de mauvaise absorption.",
            "Vérifiez si un complexe de vitamines B complet est approprié pour votre situation.",
        ],
        "Vitamin A deficiency": [
            "Discutez d'un supplément de vitamine A avec votre professionnel de santé.",
            "Demandez si une formulation bêta-carotène ou rétinol est la meilleure pour vous.",
            "Évitez des doses élevées de vitamine A sans surveillance médicale.",
        ],
        "Vitamin C deficiency": [
            "Discutez de suppléments d'acide ascorbique avec un clinicien.",
            "Envisagez une formule de vitamine C tamponnée si vous avez une sensibilité gastrique.",
            "Demandez à votre prestataire si une dose quotidienne de 500 mg à 1000 mg est appropriée.",
        ],
        "No strong visual deficiency signal": [
            "Aucun supplément spécifique n'est recommandé uniquement sur la base de l'image.",
            "Concentrez-vous sur une alimentation équilibrée et parlez à un clinicien avant de prendre des suppléments.",
            "Utilisez des tests de laboratoire pour décider si des suppléments sont nécessaires.",
        ],
    },
}

app = Flask(
    __name__,
    template_folder=str(BASE_DIR),
    static_folder=str(BASE_DIR),
    static_url_path="/static",
)
app.secret_key = os.environ.get("SECRET_KEY", "super-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)
EYE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml")
)


@dataclass
class PredictionResult:
    label: str
    confidence: int
    summary: str
    indicators: list[str]
    recommendations: list[str]
    medicines: list[str]
    metrics: dict[str, float]
    root_cause: str
    solutions: list[str]
    patient_info: dict[str, str]


def translate_ui(key: str, language: str) -> str:
    return TRANSLATIONS.get(language, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


def build_ui(language: str) -> dict[str, str]:
    return {key: translate_ui(key, language) for key in TRANSLATIONS["en"].keys()}


def translate_label(label: str, language: str) -> str:
    return LABEL_TRANSLATIONS.get(language, LABEL_TRANSLATIONS["en"]).get(label, label)


def translate_indicators(indicators: list[str], language: str) -> list[str]:
    translations = INDICATOR_TRANSLATIONS.get(language, INDICATOR_TRANSLATIONS["en"])
    return [translations.get(item, item) for item in indicators]


def translate_summary(label: str, language: str) -> str:
    return SUMMARIES.get(language, SUMMARIES["en"]).get(label, label)


def translate_recommendations(label: str, language: str) -> list[str]:
    return ADVICE_TEXTS.get(language, ADVICE_TEXTS["en"]).get(label, ADVICE_TEXTS["en"].get(label, []))


def translate_medicines(label: str, language: str) -> list[str]:
    return MEDICINE_TEXTS.get(language, MEDICINE_TEXTS["en"]).get(label, MEDICINE_TEXTS["en"].get(label, []))


def build_ui_result(result: PredictionResult, language: str) -> PredictionResult:
    return PredictionResult(
        label=translate_label(result.label, language),
        confidence=result.confidence,
        summary=translate_summary(result.label, language),
        indicators=translate_indicators(result.indicators, language),
        recommendations=translate_recommendations(result.label, language),
        medicines=translate_medicines(result.label, language),
        metrics=result.metrics,
        root_cause=result.root_cause,
        solutions=result.solutions,
        patient_info=result.patient_info,
    )


def load_report_index() -> list[dict]:
    if not METADATA_FILE.exists():
        return []
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_report_index(reports: list[dict]) -> None:
    METADATA_FILE.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


def append_report_record(record: dict) -> None:
    reports = load_report_index()
    reports.append(record)
    save_report_index(reports)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file_storage) -> Path:
    extension = file_storage.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    target = UPLOAD_DIR / filename
    file_storage.save(target)
    return target


def generate_report_file(result: PredictionResult, language: str) -> str:
    text_lines = [
        translate_ui("report_title", language),
        "",
        f"{translate_ui('prediction', language)}: {result.label}",
        f"{translate_ui('confidence', language)}: {result.confidence}%",
        "",
        result.summary,
        "",
        f"{translate_ui('patient_details', language)}:",
    ]

    for key, value in result.patient_info.items():
        if value:
            text_lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    text_lines.extend(["", f"{translate_ui('root_cause', language)}:", result.root_cause, "", f"{translate_ui('solutions', language)}:"])
    text_lines.extend(f"- {item}" for item in result.solutions)
    text_lines.extend(["", f"{translate_ui('medicines', language)}:"])
    text_lines.extend(f"- {item}" for item in result.medicines)
    text_lines.extend(["", f"{translate_ui('visual_indicators', language)}:"])
    text_lines.extend(f"- {item}" for item in result.indicators)
    text_lines.extend(["", f"{translate_ui('recommendations', language)}:"])
    text_lines.extend(f"- {item}" for item in result.recommendations)
    text_lines.extend(["", f"{translate_ui('extracted_metrics', language)}:"])
    metric_labels = METRIC_LABELS.get(language, METRIC_LABELS["en"])
    text_lines.extend(
        f"- {metric_labels.get(key, key.replace('_', ' ').title())}: {value:.2f}"
        for key, value in result.metrics.items()
    )

    filename = f"report-{uuid.uuid4().hex}.txt"
    target = REPORT_DIR / filename
    target.write_text("\n".join(text_lines), encoding="utf-8")
    return filename


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        return np.array(rgb)


def crop_interest_region(rgb_image: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return rgb_image

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    face = rgb_image[y : y + h, x : x + w]

    eyes = EYE_CASCADE.detectMultiScale(gray[y : y + h, x : x + w], scaleFactor=1.1, minNeighbors=6)
    if len(eyes) > 0:
        return face
    return face


def extract_metrics(rgb_image: np.ndarray) -> dict[str, float]:
    region = crop_interest_region(rgb_image)
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    red_channel = region[:, :, 0].astype(np.float32)
    green_channel = region[:, :, 1].astype(np.float32)
    blue_channel = region[:, :, 2].astype(np.float32)

    red_ratio = float(np.mean(red_channel) / (np.mean(gray) + 1e-6))
    yellow_ratio = float((np.mean(red_channel) + np.mean(green_channel)) / (2 * np.mean(blue_channel) + 1e-6))
    saturation = float(np.mean(hsv[:, :, 1]))

    center = region[region.shape[0] // 4 : region.shape[0] * 3 // 4, region.shape[1] // 4 : region.shape[1] * 3 // 4]
    edge = np.concatenate(
        [
            region[: region.shape[0] // 6, :, :].reshape(-1, 3),
            region[-region.shape[0] // 6 :, :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    pallor_index = float(np.mean(center) - np.mean(edge))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "red_ratio": red_ratio,
        "yellow_ratio": yellow_ratio,
        "saturation": saturation,
        "pallor_index": pallor_index,
    }


def classify_deficiency(metrics: dict[str, float]) -> PredictionResult:
    scores = {
        "Iron deficiency / anemia": 0.0,
        "Vitamin B12 deficiency": 0.0,
        "Vitamin A deficiency": 0.0,
        "Vitamin C deficiency": 0.0,
        "No strong visual deficiency signal": 0.0,
    }
    indicators: list[str] = []

    if metrics["brightness"] > 150 and metrics["red_ratio"] < 1.02:
        scores["Iron deficiency / anemia"] += 0.52
        indicators.append("High facial brightness with reduced red tone suggests pallor.")
    if metrics["pallor_index"] > 7:
        scores["Iron deficiency / anemia"] += 0.24
        indicators.append("Central facial area appears lighter than surrounding region.")

    if metrics["red_ratio"] > 1.10 and metrics["contrast"] > 45:
        scores["Vitamin B12 deficiency"] += 0.48
        indicators.append("Higher red emphasis and contrast may align with inflamed mouth or tongue regions.")
    if metrics["saturation"] > 95:
        scores["Vitamin B12 deficiency"] += 0.18

    if metrics["contrast"] < 32 and metrics["saturation"] < 72:
        scores["Vitamin A deficiency"] += 0.45
        indicators.append("Lower contrast and lower saturation can reflect dull or dry-looking skin.")
    if metrics["brightness"] < 105:
        scores["Vitamin A deficiency"] += 0.16

    if metrics["yellow_ratio"] > 1.62 and metrics["contrast"] > 38:
        scores["Vitamin C deficiency"] += 0.41
        indicators.append("Yellow-red imbalance may indicate gum or skin irritation patterns.")
    if 1.02 <= metrics["red_ratio"] <= 1.10 and metrics["saturation"] > 88:
        scores["Vitamin C deficiency"] += 0.17

    scores["No strong visual deficiency signal"] = 0.35
    if metrics["contrast"] > 36 and 1.00 <= metrics["red_ratio"] <= 1.08 and metrics["saturation"] >= 70:
        scores["No strong visual deficiency signal"] += 0.25
    if metrics["brightness"] < 150 and metrics["brightness"] > 95:
        scores["No strong visual deficiency signal"] += 0.15

    label, raw_score = max(scores.items(), key=lambda item: item[1])
    confidence = max(55, min(94, int(raw_score * 100)))

    summaries = {
        "Iron deficiency / anemia": "The image shows pallor-like patterns that can be associated with low iron or anemia.",
        "Vitamin B12 deficiency": "The image shows red and inflamed color patterns that may align with Vitamin B12-related signs.",
        "Vitamin A deficiency": "The image looks lower in contrast and saturation, which can match dry or dull visual symptoms.",
        "Vitamin C deficiency": "The color balance suggests irritation-related patterns sometimes seen with low Vitamin C.",
        "No strong visual deficiency signal": "The uploaded image does not show a strong match to the built-in visual deficiency heuristics.",
    }

    advice = {
        "Iron deficiency / anemia": [
            "Consider iron-rich foods such as spinach, beans, red meat, and lentils.",
            "Pair plant-based iron sources with Vitamin C-rich foods to improve absorption.",
            "Seek a blood test before taking supplements.",
        ],
        "Vitamin B12 deficiency": [
            "Discuss B12 testing with a clinician, especially if fatigue or numbness is present.",
            "Common sources include eggs, dairy, fish, and fortified cereals.",
            "Avoid self-diagnosing from images alone.",
        ],
        "Vitamin A deficiency": [
            "Include carrots, sweet potatoes, leafy greens, and eggs in your diet.",
            "If night-vision issues or severe dryness are present, seek medical advice.",
            "A photo cannot confirm deficiency on its own.",
        ],
        "Vitamin C deficiency": [
            "Add citrus fruits, berries, tomatoes, and peppers to meals.",
            "Persistent gum bleeding or bruising should be evaluated clinically.",
            "Lab confirmation is recommended before supplementation.",
        ],
        "No strong visual deficiency signal": [
            "The image does not strongly suggest deficiency, but symptoms still matter.",
            "Maintain a balanced diet with fruits, vegetables, protein, and hydration.",
            "Use lab tests for reliable confirmation.",
        ],
    }

    medicines = {
        "Iron deficiency / anemia": [
            "Discuss iron supplements such as ferrous sulfate with a clinician.",
            "Consider a multivitamin with iron if recommended by a doctor.",
            "Ask your provider about iron formulations that minimize stomach upset.",
        ],
        "Vitamin B12 deficiency": [
            "Discuss cyanocobalamin or methylcobalamin supplementation with a clinician.",
            "B12 injections may be recommended if absorption is impaired.",
            "Check if a complete B-complex is appropriate for your situation.",
        ],
        "Vitamin A deficiency": [
            "Discuss a vitamin A supplement with your healthcare provider.",
            "Ask if a beta-carotene or retinol formulation is best for you.",
            "Avoid high-dose vitamin A without medical supervision.",
        ],
        "Vitamin C deficiency": [
            "Discuss ascorbic acid supplements with a clinician.",
            "Consider a buffered vitamin C formula if you have stomach sensitivity.",
            "Ask your provider whether a daily 500mg to 1000mg dose is appropriate.",
        ],
        "No strong visual deficiency signal": [
            "No specific supplement is recommended based on the image alone.",
            "Focus on a balanced diet and speak with a clinician before supplementing.",
            "Use lab tests to decide whether any supplements are necessary.",
        ],
    }

    if not indicators:
        indicators.append("The analysis did not find a strong visual abnormality in the extracted image region.")

    root_causes = {
        "Iron deficiency / anemia": "Possible low dietary iron intake or absorption issues.",
        "Vitamin B12 deficiency": "Possible inadequate vitamin B12 intake or absorption.",
        "Vitamin A deficiency": "Possible insufficient vitamin A from diet or poor nutrition.",
        "Vitamin C deficiency": "Possible low vitamin C intake or poor dietary balance.",
        "No strong visual deficiency signal": "No strong visual nutrition deficiency signal detected.",
    }

    solutions = {
        "Iron deficiency / anemia": [
            "Increase iron-rich foods such as spinach, beans, red meat, and lentils.",
            "Combine iron sources with vitamin C-rich foods to improve absorption.",
            "Consult a clinician for blood tests before taking supplements.",
        ],
        "Vitamin B12 deficiency": [
            "Include eggs, dairy, fish, and fortified cereals in the diet.",
            "Monitor neurological symptoms such as tingling or fatigue.",
            "Avoid self-diagnosis from an image alone.",
        ],
        "Vitamin A deficiency": [
            "Add carrots, sweet potatoes, leafy greens, and eggs to meals.",
            "If dry skin or vision changes persist, seek medical advice.",
            "Maintain a balanced diet with colorful vegetables.",
        ],
        "Vitamin C deficiency": [
            "Add citrus fruits, berries, tomatoes, and peppers to your diet.",
            "Watch for persistent gum bleeding or easy bruising.",
            "Use laboratory tests for accurate confirmation.",
        ],
        "No strong visual deficiency signal": [
            "Keep a balanced diet with vegetables, fruits, protein, and hydration.",
            "Monitor symptoms and consult a clinician if concerned.",
            "Use proper tests rather than relying on images alone.",
        ],
    }

    return PredictionResult(
        label=label,
        confidence=confidence,
        summary=summaries[label],
        indicators=indicators,
        recommendations=advice[label],
        medicines=medicines[label],
        metrics=metrics,
        root_cause=root_causes[label],
        solutions=solutions[label],
        patient_info={},
    )


@app.route("/", methods=["GET", "POST"])
def index():
    language = request.form.get("language", request.args.get("language", "en"))
    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    ui = build_ui(language)
    metric_labels = METRIC_LABELS.get(language, METRIC_LABELS["en"])

    if request.method == "POST":
        file = request.files.get("image")
        patient_info = {
            "patient_name": request.form.get("patient_name", "").strip(),
            "patient_age": request.form.get("patient_age", "").strip(),
            "patient_gender": request.form.get("patient_gender", "").strip(),
            "patient_notes": request.form.get("patient_notes", "").strip(),
        }

        if not file or not file.filename:
            return render_template(
                "index.html",
                ui=ui,
                languages=SUPPORTED_LANGUAGES,
                language=language,
                error=translate_ui("error_no_image", language),
                metric_labels=metric_labels,
            )
        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                ui=ui,
                languages=SUPPORTED_LANGUAGES,
                language=language,
                error=translate_ui("error_invalid", language),
                metric_labels=metric_labels,
            )

        image_path = save_upload(file)
        metrics = extract_metrics(load_image(image_path))
        prediction = classify_deficiency(metrics)
        prediction.patient_info = patient_info
        localized_result = build_ui_result(prediction, language)
        report_filename = generate_report_file(localized_result, language)
        image_url = url_for("static_upload", filename=image_path.name)
        report_url = url_for("download_report", filename=report_filename)

        append_report_record(
            {
                "id": report_filename,
                "timestamp": datetime.now().isoformat(),
                "language": language,
                "label": localized_result.label,
                "confidence": localized_result.confidence,
                "summary": localized_result.summary,
                "root_cause": localized_result.root_cause,
                "solutions": localized_result.solutions,
                "patient_info": localized_result.patient_info,
                "image_name": image_path.name,
                "report_name": report_filename,
            }
        )

        return render_template(
            "index.html",
            ui=ui,
            languages=SUPPORTED_LANGUAGES,
            language=language,
            result=localized_result,
            image_url=image_url,
            report_url=report_url,
            metric_labels=metric_labels,
        )

    return render_template(
        "index.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        metric_labels=metric_labels,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    error = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == os.environ.get("ADMIN_USER", "admin") and password == os.environ.get("ADMIN_PASS", "admin123"):
            session["logged_in"] = True
            return redirect(url_for("admin"))
        error = "Invalid credentials"

    return render_template(
        "login.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        error=error,
    )


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))


@app.route("/admin")
def admin():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    metric_labels = METRIC_LABELS.get(language, METRIC_LABELS["en"])
    reports = sorted(load_report_index(), key=lambda r: r.get("timestamp", ""), reverse=True)
    uploads = [p.name for p in UPLOAD_DIR.iterdir() if p.is_file()]
    ports = [port.strip() for port in os.environ.get("MULTI_PORTS", "5000,5001,5002").split(",") if port.strip()]
    current_port = os.environ.get("PORT", "5000")

    return render_template(
        "admin.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        reports=reports,
        uploads=uploads,
        ports=ports,
        current_port=current_port,
        metric_labels=metric_labels,
    )


@app.route("/reports")
def reports_page():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    reports = sorted(load_report_index(), key=lambda r: r.get("timestamp", ""), reverse=True)
    return render_template(
        "reports.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        reports=reports,
    )


@app.route("/uploaded-images")
def uploaded_images():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    ui = build_ui(language)
    images = [p.name for p in UPLOAD_DIR.iterdir() if p.is_file()]
    return render_template(
        "uploads.html",
        ui=ui,
        languages=SUPPORTED_LANGUAGES,
        language=language,
        images=images,
    )


@app.route("/reports/<path:filename>")
def download_report(filename: str):
    return send_from_directory(str(REPORT_DIR), filename, as_attachment=True)


@app.route("/uploads/<path:filename>")
def static_upload(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/debug-vitscan")
def debug_vitscan():
    language = request.args.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    return {
        "base_dir": str(BASE_DIR),
        "template_folder": app.template_folder,
        "static_folder": app.static_folder,
        "ui_title": build_ui(language)["title"],
        "translation_title": TRANSLATIONS[language]["title"],
        "index_exists": (BASE_DIR / "index.html").exists(),
    }


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=debug)

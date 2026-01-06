"""
Diabetes Risk Prediction App for Older Adults
==============================================
An elderly-friendly interface for diabetes risk assessment
with explainable AI (SHAP) integration.

Designed following accessibility guidelines for older adults based on research:
- Large fonts (18-24px minimum) and buttons (minimum 44x44px touch targets)
- High contrast colors (WCAG AAA compliance, 7:1 ratio minimum)
- Progressive disclosure (one section at a time)
- Simple navigation with clear visual hierarchy
- Clear explanations with minimal cognitive load
- Tooltips and contextual help
- Progress indicators
- Larger spacing between elements
- Error prevention with validation

References:
- Nielsen Norman Group: "Usability for Senior Citizens"
- W3C Web Accessibility Guidelines (WCAG 2.1 AAA)
- "Design Guidelines for Elderly Users" - International Journal of HCI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import time

# ============================================================================
# TRANSLATIONS - English and Malay
# ============================================================================
TRANSLATIONS = {
    'en': {
        'welcome_title': 'Welcome to Your Health Assessment!',
        'welcome_msg': 'This simple tool helps you understand your likelihood of developing diabetes.',
        'easy_steps': 'Easy step-by-step questions',
        'clear_results': 'Clear, personalized results',
        'understand_factors': 'Understand your health factors',
        'get_recommendations': 'Get helpful recommendations',
        'step': 'Step',
        'of': 'of',
        'need_help': 'Need Help?',
        'help_info': 'Help Information',
        'next': 'NEXT',
        'back': 'BACK',
        'calculate': 'CALCULATE MY PROBABILITY',
        'start_over': 'START OVER',
        'confirm_restart_title': 'Confirm Restart',
        'confirm_restart_msg': 'Are you sure you want to start over? All your answers will be lost.',
        'yes': 'Yes',
        'no': 'No',
        
        # Step 1
        'step1_title': 'Tell Us About Yourself',
        'step1_help': 'We need some basic information to assess your diabetes probability accurately. All information is kept private and used only for this assessment.',
        'age_label': 'What is your age?',
        'sex_label': 'What is your biological sex?',
        'female': 'Female',
        'male': 'Male',
        'education_label': 'What is your highest education level?',
        'employment_label': 'What is your employment status?',
        
        # Step 2
        'step2_title': 'Your Physical Health',
        'step2_help': 'These measurements help us understand your overall physical health. BMI will be calculated automatically from your height and weight.',
        'weight_label': 'What is your weight (in kg)?',
        'height_label': 'What is your height (in cm)?',
        'bmi_calculated': 'Your Body Mass Index (BMI)',
        'gen_health_label': 'How would you rate your general health?',
        'checkup_label': 'When was your last medical checkup?',
        
        # Step 3
        'step3_title': 'Health Conditions & Medications',
        'step3_help': 'These questions help us understand existing health conditions that may affect your diabetes probability. Answer based on what your doctor has told you.',
        'bp_meds_label': 'Do you currently take medication for HIGH BLOOD PRESSURE?',
        'chol_meds_label': 'Do you currently take medication for HIGH CHOLESTEROL?',
        'doctor_visits_label': 'How often do you visit a doctor?',
        'yes_bp': 'Yes, I take BP medication',
        'no_bp': "No, I don't take BP medication",
        'yes_chol': 'Yes, I take cholesterol medication',
        'no_chol': "No, I don't take cholesterol medication",
        
        # Step 4
        'step4_title': 'Your Lifestyle Habits',
        'step4_help': 'Your daily habits have a big impact on your diabetes probability. Be honest - this helps us give you the most accurate assessment.',
        'exercise_label': 'Do you exercise or do physical activity regularly?',
        'alcohol_label': 'How would you describe your alcohol consumption?',
        'yes_exercise': 'Yes, I exercise regularly (at least 30 min, 3+ times/week)',
        'no_exercise': "No, I don't exercise regularly",
        
        # Step 5 - Results
        'results_title': 'Your Personalized Results',
        'analyzing': 'Analyzing your health data...',
        'diabetes_likelihood': 'Your Diabetes Likelihood',
        'probability_score': 'Probability Score',
        'current_profile': 'This means your current health profile shows a',
        'likelihood_of': 'likelihood of developing diabetes.',
        
        'low_title': 'Good News!',
        'low_msg': 'Your current health profile suggests a **lower likelihood** of developing diabetes.',
        'low_keep': 'Keep up the good work by:',
        'low_1': 'Maintaining your healthy diet',
        'low_2': 'Continuing to stay physically active',
        'low_3': 'Getting regular health check-ups',
        'low_4': 'Monitoring your weight and BMI',
        'low_remember': "Even with low likelihood, it's important to maintain healthy habits!",
        
        'moderate_title': 'Attention Needed',
        'moderate_msg': 'Your health profile shows **some factors** for diabetes that need attention.',
        'moderate_what': 'What you should do:',
        'moderate_1': '**Talk to your doctor** about diabetes prevention',
        'moderate_2': 'Review and improve your diet',
        'moderate_3': 'Increase physical activity (aim for 30 min/day, 5 days/week)',
        'moderate_4': 'Monitor your blood sugar levels',
        'moderate_5': 'Work towards a healthier weight if needed',
        'moderate_good': 'Lifestyle changes can significantly improve your health!',
        
        'high_title': 'Important - Action Required!',
        'high_msg': 'Your health profile indicates **higher likelihood** for diabetes.',
        'high_steps': 'Take these important steps:',
        'high_1': "**Schedule a doctor appointment SOON** - Don't delay!",
        'high_2': 'Get a comprehensive diabetes screening (blood glucose test)',
        'high_3': 'Discuss a personalized prevention plan with your doctor',
        'high_4': 'Make immediate dietary improvements',
        'high_5': 'Start increasing your physical activity gradually',
        'high_6': 'Work on weight management with medical guidance',
        'high_remember': 'Early action can prevent or delay diabetes onset!',
        
        'understanding_title': 'Understanding Your Results',
        'what_influences': 'What influences your result?',
        'influences_msg': 'Below are the main health factors that affected your diabetes probability score. Understanding these can help you make better health decisions.',
        'top5_factors': 'Top 5 Factors Affecting Your Results:',
        'increases_prob': 'Increases Probability',
        'decreases_prob': 'Decreases Probability',
        'impact': 'Impact: This factor',
        'your_prob': 'your diabetes probability.',
        
        'action_plan_title': 'Your Personalized Action Plan',
        
        # Recommendations
        'rec_weight_title': 'Weight Management',
        'rec_weight_text': 'Your BMI suggests working towards a healthier weight. Even losing 5-7% of your body weight can significantly improve your health.',
        'rec_weight_action': 'Talk to your doctor about safe weight loss strategies',
        'rec_exercise_title': 'Physical Activity',
        'rec_exercise_text': 'Regular exercise is one of the best ways to stay healthy and prevent diabetes.',
        'rec_exercise_action': 'Start with 10-15 minutes of walking daily, then gradually increase to 30 minutes, 5 days per week',
        'rec_health_title': 'Overall Health',
        'rec_health_text': 'Your general health rating suggests room for improvement.',
        'rec_health_action': 'Schedule a comprehensive health checkup with your doctor to address any concerns',
        'rec_meds_title': 'Medication Management',
        'rec_meds_text': 'You are taking medications for blood pressure or cholesterol.',
        'rec_meds_action': 'Keep taking your medications as prescribed and attend regular follow-ups',
        'rec_alcohol_title': 'Alcohol Consumption',
        'rec_alcohol_text': 'High alcohol consumption can affect your health.',
        'rec_alcohol_action': 'Consider reducing alcohol intake - talk to your doctor about safe limits',
        'rec_good_title': 'Keep Up the Good Work!',
        'rec_good_text': 'Your current health habits are positive.',
        'rec_good_action': 'Continue your healthy lifestyle and maintain regular health checkups',
        'action_step': 'Action Step:',
        
        # Next steps
        'next_steps_title': 'What To Do Next',
        'recommended_steps': 'Recommended Next Steps:',
        'next_1': 'Save or print these results to discuss with your doctor',
        'next_2': 'Schedule a doctor appointment for a proper diabetes screening',
        'next_3': 'Start one healthy change today - even small steps matter!',
        'next_4': 'Re-assess in 3-6 months to track your progress',
        'next_5': 'Learn more about diabetes prevention from reliable sources',
        'print_tip': 'Tip:',
        'print_msg': "Use your browser's print function (Ctrl+P or Cmd+P) to save or print these results for your doctor.",
        
        # Medical disclaimer
        'medical_disclaimer_title': 'Important Medical Disclaimer',
        'disclaimer_msg': 'This tool is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.',
        'disclaimer_1': 'Always seek the advice of your physician or qualified health provider with any questions about your health',
        'disclaimer_2': 'Never disregard professional medical advice or delay seeking it because of something you learned from this tool',
        'disclaimer_3': 'If you think you may have a medical emergency, call your doctor or emergency services immediately',
        'disclaimer_4': 'This tool uses statistical models and may not account for all individual health factors',
        
        'significantly': 'significantly',
        'slightly': 'slightly',
        'increases': 'increases',
        'decreases': 'decreases',
        'LOW': 'LOW',
        'MODERATE': 'MODERATE',
        'HIGH': 'HIGH',
        'lower': 'lower',
        'moderate': 'moderate',
        'higher': 'higher',
    },
    'ms': {
        'welcome_title': 'Selamat Datang ke Penilaian Kesihatan Anda!',
        'welcome_msg': 'Alat mudah ini membantu anda memahami kemungkinan mendapat diabetes.',
        'easy_steps': 'Soalan langkah demi langkah yang mudah',
        'clear_results': 'Keputusan yang jelas dan diperibadikan',
        'understand_factors': 'Fahami faktor kesihatan anda',
        'get_recommendations': 'Dapatkan cadangan yang berguna',
        'step': 'Langkah',
        'of': 'daripada',
        'need_help': 'Perlukan Bantuan?',
        'help_info': 'Maklumat Bantuan',
        'next': 'SETERUSNYA',
        'back': 'KEMBALI',
        'calculate': 'KIRA KEBARANGKALIAN SAYA',
        'start_over': 'MULA SEMULA',
        'confirm_restart_title': 'Sahkan Mula Semula',
        'confirm_restart_msg': 'Adakah anda pasti mahu mula semula? Semua jawapan anda akan hilang.',
        'yes': 'Ya',
        'no': 'Tidak',
        
        # Step 1
        'step1_title': 'Beritahu Kami Tentang Diri Anda',
        'step1_help': 'Kami memerlukan maklumat asas untuk menilai kebarangkalian diabetes anda dengan tepat. Semua maklumat adalah sulit dan hanya digunakan untuk penilaian ini.',
        'age_label': 'Berapakah umur anda?',
        'sex_label': 'Apakah jantina biologi anda?',
        'female': 'Perempuan',
        'male': 'Lelaki',
        'education_label': 'Apakah tahap pendidikan tertinggi anda?',
        'employment_label': 'Apakah status pekerjaan anda?',
        
        # Step 2
        'step2_title': 'Kesihatan Fizikal Anda',
        'step2_help': 'Pengukuran ini membantu kami memahami kesihatan fizikal keseluruhan anda. BMI akan dikira secara automatik daripada ketinggian dan berat badan anda.',
        'weight_label': 'Berapakah berat badan anda (dalam kg)?',
        'height_label': 'Berapakah ketinggian anda (dalam cm)?',
        'bmi_calculated': 'Indeks Jisim Badan (BMI) Anda',
        'gen_health_label': 'Bagaimanakah anda menilai kesihatan am anda?',
        'checkup_label': 'Bilakah pemeriksaan perubatan terakhir anda?',
        
        # Step 3
        'step3_title': 'Keadaan Kesihatan & Ubat-ubatan',
        'step3_help': 'Soalan ini membantu kami memahami keadaan kesihatan sedia ada yang mungkin mempengaruhi kebarangkalian diabetes anda. Jawab berdasarkan apa yang doktor anda beritahu.',
        'bp_meds_label': 'Adakah anda mengambil ubat untuk TEKANAN DARAH TINGGI?',
        'chol_meds_label': 'Adakah anda mengambil ubat untuk KOLESTEROL TINGGI?',
        'doctor_visits_label': 'Berapa kerap anda melawat doktor?',
        'yes_bp': 'Ya, saya mengambil ubat tekanan darah',
        'no_bp': 'Tidak, saya tidak mengambil ubat tekanan darah',
        'yes_chol': 'Ya, saya mengambil ubat kolesterol',
        'no_chol': 'Tidak, saya tidak mengambil ubat kolesterol',
        
        # Step 4
        'step4_title': 'Tabiat Gaya Hidup Anda',
        'step4_help': 'Tabiat harian anda memberi kesan besar kepada kebarangkalian diabetes anda. Bersikap jujur - ini membantu kami memberi penilaian yang paling tepat.',
        'exercise_label': 'Adakah anda bersenam atau melakukan aktiviti fizikal dengan kerap?',
        'alcohol_label': 'Bagaimanakah anda menggambarkan pengambilan alkohol anda?',
        'yes_exercise': 'Ya, saya bersenam dengan kerap (sekurang-kurangnya 30 min, 3+ kali/minggu)',
        'no_exercise': 'Tidak, saya tidak bersenam dengan kerap',
        
        # Step 5 - Results
        'results_title': 'Keputusan Diperibadikan Anda',
        'analyzing': 'Menganalisis data kesihatan anda...',
        'diabetes_likelihood': 'Kemungkinan Diabetes Anda',
        'probability_score': 'Skor Kebarangkalian',
        'current_profile': 'Ini bermakna profil kesihatan semasa anda menunjukkan',
        'likelihood_of': 'kemungkinan mendapat diabetes.',
        
        'low_title': 'Berita Baik!',
        'low_msg': 'Profil kesihatan semasa anda menunjukkan **kemungkinan lebih rendah** untuk mendapat diabetes.',
        'low_keep': 'Teruskan usaha baik dengan:',
        'low_1': 'Mengekalkan diet sihat anda',
        'low_2': 'Terus aktif secara fizikal',
        'low_3': 'Mendapat pemeriksaan kesihatan berkala',
        'low_4': 'Memantau berat badan dan BMI anda',
        'low_remember': 'Walaupun kemungkinan rendah, penting untuk mengekalkan tabiat sihat!',
        
        'moderate_title': 'Perhatian Diperlukan',
        'moderate_msg': 'Profil kesihatan anda menunjukkan **beberapa faktor** untuk diabetes yang memerlukan perhatian.',
        'moderate_what': 'Apa yang perlu anda lakukan:',
        'moderate_1': '**Berjumpa dengan doktor** tentang pencegahan diabetes',
        'moderate_2': 'Kaji semula dan perbaiki diet anda',
        'moderate_3': 'Tingkatkan aktiviti fizikal (sasaran 30 min/hari, 5 hari/minggu)',
        'moderate_4': 'Pantau paras gula dalam darah anda',
        'moderate_5': 'Usaha ke arah berat badan yang lebih sihat jika perlu',
        'moderate_good': 'Perubahan gaya hidup boleh meningkatkan kesihatan anda dengan ketara!',
        
        'high_title': 'Penting - Tindakan Diperlukan!',
        'high_msg': 'Profil kesihatan anda menunjukkan **kemungkinan lebih tinggi** untuk diabetes.',
        'high_steps': 'Ambil langkah penting ini:',
        'high_1': '**Jadualkan temujanji doktor SEGERA** - Jangan bertangguh!',
        'high_2': 'Dapatkan saringan diabetes menyeluruh (ujian glukosa darah)',
        'high_3': 'Bincangkan pelan pencegahan diperibadikan dengan doktor anda',
        'high_4': 'Buat penambahbaikan diet segera',
        'high_5': 'Mula tingkatkan aktiviti fizikal anda secara beransur',
        'high_6': 'Usaha pengurusan berat badan dengan bimbingan perubatan',
        'high_remember': 'Tindakan awal boleh mencegah atau menangguhkan permulaan diabetes!',
        
        'understanding_title': 'Memahami Keputusan Anda',
        'what_influences': 'Apa yang mempengaruhi keputusan anda?',
        'influences_msg': 'Di bawah adalah faktor kesihatan utama yang mempengaruhi skor kebarangkalian diabetes anda. Memahami ini boleh membantu anda membuat keputusan kesihatan yang lebih baik.',
        'top5_factors': '5 Faktor Teratas yang Mempengaruhi Keputusan Anda:',
        'increases_prob': 'Meningkatkan Kebarangkalian',
        'decreases_prob': 'Mengurangkan Kebarangkalian',
        'impact': 'Kesan: Faktor ini',
        'your_prob': 'kebarangkalian diabetes anda.',
        
        'action_plan_title': 'Pelan Tindakan Diperibadikan Anda',
        
        # Recommendations  
        'rec_weight_title': 'Pengurusan Berat Badan',
        'rec_weight_text': 'BMI anda menunjukkan perlu berusaha ke arah berat badan yang lebih sihat. Walaupun kehilangan 5-7% daripada berat badan anda boleh meningkatkan kesihatan anda dengan ketara.',
        'rec_weight_action': 'Berjumpa dengan doktor tentang strategi penurunan berat badan yang selamat',
        'rec_exercise_title': 'Aktiviti Fizikal',
        'rec_exercise_text': 'Senaman berkala adalah salah satu cara terbaik untuk kekal sihat dan mencegah diabetes.',
        'rec_exercise_action': 'Mulakan dengan 10-15 minit berjalan kaki setiap hari, kemudian tingkatkan secara beransur kepada 30 minit, 5 hari seminggu',
        'rec_health_title': 'Kesihatan Keseluruhan',
        'rec_health_text': 'Penilaian kesihatan am anda menunjukkan ruang untuk penambahbaikan.',
        'rec_health_action': 'Jadualkan pemeriksaan kesihatan menyeluruh dengan doktor anda untuk menangani sebarang kebimbangan',
        'rec_meds_title': 'Pengurusan Ubat-ubatan',
        'rec_meds_text': 'Anda mengambil ubat untuk tekanan darah atau kolesterol.',
        'rec_meds_action': 'Terus ambil ubat anda seperti yang ditetapkan dan hadiri susulan berkala',
        'rec_alcohol_title': 'Pengambilan Alkohol',
        'rec_alcohol_text': 'Pengambilan alkohol yang tinggi boleh menjejaskan kesihatan anda.',
        'rec_alcohol_action': 'Pertimbangkan untuk mengurangkan pengambilan alkohol - bincang dengan doktor tentang had yang selamat',
        'rec_good_title': 'Teruskan Usaha Baik!',
        'rec_good_text': 'Tabiat kesihatan semasa anda adalah positif.',
        'rec_good_action': 'Teruskan gaya hidup sihat anda dan kekalkan pemeriksaan kesihatan berkala',
        'action_step': 'Langkah Tindakan:',
        
        # Next steps
        'next_steps_title': 'Apa Yang Perlu Dilakukan Seterusnya',
        'recommended_steps': 'Langkah Seterusnya yang Disyorkan:',
        'next_1': 'Simpan atau cetak keputusan ini untuk dibincangkan dengan doktor anda',
        'next_2': 'Jadualkan temujanji doktor untuk saringan diabetes yang betul',
        'next_3': 'Mulakan satu perubahan sihat hari ini - walaupun langkah kecil penting!',
        'next_4': 'Nilai semula dalam 3-6 bulan untuk menjejaki kemajuan anda',
        'next_5': 'Ketahui lebih lanjut tentang pencegahan diabetes daripada sumber yang boleh dipercayai',
        'print_tip': 'Petua:',
        'print_msg': 'Gunakan fungsi cetak penyemak imbas anda (Ctrl+P atau Cmd+P) untuk menyimpan atau mencetak keputusan ini untuk doktor anda.',
        
        # Medical disclaimer
        'medical_disclaimer_title': 'Penafian Perubatan Penting',
        'disclaimer_msg': 'Alat ini adalah untuk tujuan pendidikan sahaja dan bukan pengganti nasihat perubatan profesional, diagnosis, atau rawatan.',
        'disclaimer_1': 'Sentiasa dapatkan nasihat doktor anda atau penyedia kesihatan berkelayakan untuk sebarang soalan tentang kesihatan anda',
        'disclaimer_2': 'Jangan abaikan nasihat perubatan profesional atau tangguhkan mendapatkannya kerana sesuatu yang anda pelajari daripada alat ini',
        'disclaimer_3': 'Jika anda fikir anda mungkin mengalami kecemasan perubatan, hubungi doktor atau perkhidmatan kecemasan anda segera',
        'disclaimer_4': 'Alat ini menggunakan model statistik dan mungkin tidak mengambil kira semua faktor kesihatan individu',
        
        'significantly': 'dengan ketara',
        'slightly': 'sedikit',
        'increases': 'meningkatkan',
        'decreases': 'mengurangkan',
        'LOW': 'RENDAH',
        'MODERATE': 'SEDERHANA',
        'HIGH': 'TINGGI',
        'lower': 'lebih rendah',
        'moderate': 'sederhana',
        'higher': 'lebih tinggi',
    }
}

def t(key):
    """Get translation for current language"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

# Additional translation dictionaries for form options
AGE_GROUPS = {
    'en': {1: "18-24 years", 2: "25-29 years", 3: "30-34 years", 4: "35-39 years", 5: "40-44 years", 6: "45-49 years",
           7: "50-54 years", 8: "55-59 years", 9: "60-64 years", 10: "65-69 years", 11: "70-74 years", 12: "75-79 years", 13: "80 years or older"},
    'ms': {1: "18-24 tahun", 2: "25-29 tahun", 3: "30-34 tahun", 4: "35-39 tahun", 5: "40-44 tahun", 6: "45-49 tahun",
           7: "50-54 tahun", 8: "55-59 tahun", 9: "60-64 tahun", 10: "65-69 tahun", 11: "70-74 tahun", 12: "75-79 tahun", 13: "80 tahun atau lebih"}
}
EDUCATION_LEVELS = {'en': {1: "Never attended school", 2: "Elementary school", 3: "Some high school", 4: "High school graduate", 5: "Some college or technical school", 6: "College graduate or higher"},
                    'ms': {1: "Tidak pernah bersekolah", 2: "Sekolah rendah", 3: "Sebahagian sekolah menengah", 4: "Lulus sekolah menengah", 5: "Sebahagian kolej atau sekolah teknikal", 6: "Lulus kolej atau lebih tinggi"}}
EMPLOYMENT_STATUS = {'en': {1: "Employed for wages", 2: "Self-employed", 3: "Unemployed", 4: "Retired", 5: "Unable to work", 6: "Student or homemaker"},
                     'ms': {1: "Bekerja bergaji", 2: "Bekerja sendiri", 3: "Menganggur", 4: "Bersara", 5: "Tidak dapat bekerja", 6: "Pelajar atau suri rumah"}}
HEALTH_RATING = {'en': {1: "⭐⭐⭐⭐⭐ Excellent", 2: "⭐⭐⭐⭐ Very Good", 3: "⭐⭐⭐ Good", 4: "⭐⭐ Fair", 5: "⭐ Poor"},
                 'ms': {1: "⭐⭐⭐⭐⭐ Cemerlang", 2: "⭐⭐⭐⭐ Sangat Baik", 3: "⭐⭐⭐ Baik", 4: "⭐⭐ Sederhana", 5: "⭐ Lemah"}}
CHECKUP_STATUS = {'en': {1: "Within past year", 2: "Within past 2 years", 3: "Within past 5 years", 4: "5 or more years ago", 5: "Never"},
                  'ms': {1: "Dalam tahun lepas", 2: "Dalam 2 tahun lepas", 3: "Dalam 5 tahun lepas", 4: "5 tahun atau lebih lalu", 5: "Tidak pernah"}}
DOCTOR_VISITS = {'en': {1: "Regularly (multiple times per year)", 2: "Annually (once per year)", 3: "Occasionally (every few years)", 4: "Rarely or never"},
                 'ms': {1: "Kerap (beberapa kali setahun)", 2: "Tahunan (sekali setahun)", 3: "Sekali-sekala (beberapa tahun sekali)", 4: "Jarang atau tidak pernah"}}
ALCOHOL_STATUS = {'en': {1: "Non-drinker", 2: "Light drinker (1-2 drinks per week)", 3: "Moderate drinker (3-7 drinks per week)", 4: "Heavy drinker (8+ drinks per week)"},
                  'ms': {1: "Tidak minum", 2: "Peminum ringan (1-2 minuman seminggu)", 3: "Peminum sederhana (3-7 minuman seminggu)", 4: "Peminum berat (8+ minuman seminggu)"}}

# ============================================================================
# PAGE CONFIGURATION - Elderly-Friendly Settings
# ============================================================================
st.set_page_config(
    page_title="Diabetes Probability Assessment",
    page_icon="🩺",
    layout="centered",  # Centered for better focus
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Diabetes Probability Assessment Tool for Older Adults - Educational Use Only"
    }
)

# Initialize session state for step-by-step navigation
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'show_help' not in st.session_state:
    st.session_state.show_help = {}
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # 'en' for English, 'ms' for Malay
if 'font_size' not in st.session_state:
    st.session_state.font_size = 20  # Default font size
if 'show_confirmation' not in st.session_state:
    st.session_state.show_confirmation = False
if 'screen_reader_mode' not in st.session_state:
    st.session_state.screen_reader_mode = False
if 'show_keyboard_shortcuts' not in st.session_state:
    st.session_state.show_keyboard_shortcuts = False

# ============================================================================
# CUSTOM CSS FOR ELDERLY-FRIENDLY UI
# ============================================================================

# Dynamic font size CSS based on user preference
font_size = st.session_state.get('font_size', 20)
heading_multiplier = 2.4  # 48px at default 20px
section_multiplier = 1.6  # 32px at default 20px
label_multiplier = 1.1  # 22px at default 20px

st.markdown(f"""
<style>
    /* Base font size - adjustable for elderly users (WCAG AAA) */
    html, body, [class*="css"] {{
        font-size: {font_size}px !important;
        line-height: 1.6 !important;
    }}
    
    /* Main container - centered with max width for readability */
    .main .block-container {{
        max-width: 900px;
        padding: 2rem 1.5rem;
    }}
    
    /* Main title styling - High contrast */
    .main-title {{
        font-size: {font_size * heading_multiplier}px !important;
        font-weight: bold !important;
        color: #000000 !important;
        text-align: center;
        padding: 25px 0;
        margin-bottom: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    /* Section headers - High contrast (7:1 ratio) */
    .section-header {{
        font-size: {font_size * section_multiplier}px !important;
        font-weight: bold !important;
        border-bottom: 4px solid #667eea;
        padding-bottom: 15px;
        margin: 40px 0 25px 0;
    }}
    
    /* Step indicator - Clear progress tracking */
    .step-indicator {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 60px;
        padding: 20px 40px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Large readable labels with high contrast */
    .stSelectbox label, .stSlider label, .stRadio label, .stNumberInput label {{
        font-size: {font_size * label_multiplier}px !important;
        font-weight: 700 !important;
        line-height: 1.5 !important;
        margin-bottom: 10px !important;
    }}
    
    /* Larger buttons - Minimum 44x44px touch target */
    .stButton {{
        width: 100% !important;
    }}
    
    .stButton > button {{
        font-size: {max(font_size - 4, 14)}px !important;
        font-weight: 600 !important;
        padding: 15px 20px !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 3px solid #5a67d8 !important;
        width: 100% !important;
        margin: 10px 0 !important;
        min-height: 56px !important;
        height: auto !important;
        box-sizing: border-box !important;
        overflow: visible !important;
        overflow-wrap: break-word !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.4 !important;
        display: block !important;
        text-align: center !important;
    }}
    
    /* Button text content - target all possible inner elements */
    .stButton > button > div,
    .stButton > button > div > p,
    .stButton > button > p,
    .stButton > button > span,
    .stButton > button div,
    .stButton > button p,
    .stButton > button span {{
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        hyphens: auto !important;
        text-align: center !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.4 !important;
        max-width: 100% !important;
        width: 100% !important;
        display: block !important;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, #5a67d8 0%, #6b46a0 100%) !important;
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }}
    
    .stButton > button:active {{
        transform: scale(0.98);
    }}
    
    /* Probability result boxes - Dark mode compatible */
    .prob-low {{
        background-color: color-mix(in srgb, #22C55E 15%, transparent);
        border: 5px solid #22C55E;
        border-radius: 25px;
        padding: 40px;
        text-align: center;
        margin: 25px 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }}
    
    .prob-medium {{
        background-color: color-mix(in srgb, #EAB308 15%, transparent);
        border: 5px solid #EAB308;
        border-radius: 25px;
        padding: 40px;
        text-align: center;
        margin: 25px 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }}
    
    .prob-high {{
        background-color: color-mix(in srgb, #EF4444 15%, transparent);
        border: 5px solid #EF4444;
        border-radius: 25px;
        padding: 40px;
        text-align: center;
        margin: 25px 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }}
    
    .prob-text {{
        font-size: 42px !important;
        font-weight: bold !important;
    }}
    
    .probability-text {{
        font-size: 28px !important;
        margin-top: 20px;
        font-weight: 600;
    }}
    
    /* Info boxes - High contrast with clear borders - Dark mode compatible */
    .info-box {{
        background-color: color-mix(in srgb, var(--primary-color, #667eea) 15%, transparent);
        border: 3px solid var(--primary-color, #667eea);
        border-left: 8px solid var(--primary-color, #667eea);
        padding: 25px;
        margin: 20px 0;
        border-radius: 15px;
        font-size: 20px;
        line-height: 1.8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    .help-box {{
        background-color: color-mix(in srgb, #DD6B20 10%, transparent);
        border: 3px solid #DD6B20;
        border-left: 8px solid #DD6B20;
        padding: 20px;
        margin: 15px 0;
        border-radius: 12px;
        font-size: 18px;
        line-height: 1.7;
    }}
    
    /* Action plan cards - Dark mode compatible */
    .action-card {{
        background-color: color-mix(in srgb, #3B82F6 12%, transparent);
        border: 3px solid #3B82F6;
        border-radius: 18px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        line-height: 1.7;
    }}
    
    .action-card .action-highlight {{
        color: #60A5FA !important;
    }}
    
    /* Explanation cards with better spacing - Dark mode compatible */
    .explanation-card {{
        background-color: color-mix(in srgb, currentColor 8%, transparent);
        border: 3px solid color-mix(in srgb, currentColor 30%, transparent);
        border-radius: 18px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        line-height: 1.7;
    }}
    
    /* SHAP factor cards - Dark mode compatible */
    .factor-card-positive {{
        background-color: color-mix(in srgb, #FC8181 12%, transparent);
        border: 3px solid #FC8181;
        border-left: 6px solid #FC8181;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
    }}
    
    .factor-card-negative {{
        background-color: color-mix(in srgb, #68D391 12%, transparent);
        border: 3px solid #68D391;
        border-left: 6px solid #68D391;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
    }}
    
    /* Radio buttons - Larger touch targets - Dark mode compatible */
    .stRadio > div {{
        gap: 20px !important;
    }}
    
    .stRadio > div > label {{
        padding: 15px 25px !important;
        font-size: 20px !important;
        border: 2px solid rgba(128, 128, 128, 0.3) !important;
        border-radius: 12px !important;
        min-width: 120px;
        text-align: center;
    }}
    
    .stRadio > div > label:hover {{
        border-color: #667eea !important;
    }}
    
    /* Language toggle button */
    [data-testid="stButton"][key="lang_toggle"] button,
    button[kind="secondary"] {{
        white-space: nowrap !important;
        min-width: 100px !important;
        padding: 15px 20px !important;
        font-size: 18px !important;
    }}
    
    /* Progress bar */
    .progress-bar {{
        width: 100%;
        height: 12px;
        background-color: #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
        margin: 20px 0;
    }}
    
    .progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
    }}
    
    /* Success/Warning/Error messages - High contrast */
    .stSuccess, .stWarning, .stError, .stInfo {{
        font-size: 19px !important;
        padding: 20px !important;
        border-radius: 12px !important;
        line-height: 1.7 !important;
    }}
    
    /* Footer - Dark mode compatible */
    .footer {{
        text-align: center;
        opacity: 0.7;
        font-size: 16px;
        padding: 40px 0;
        border-top: 2px solid color-mix(in srgb, currentColor 20%, transparent);
        margin-top: 60px;
    }}
    
    /* Accessibility: Focus indicators */
    *:focus {{
        outline: 3px solid #4299E1 !important;
        outline-offset: 2px !important;
    }}
    
    /* Reduce motion for users who prefer it */
    @media (prefers-reduced-motion: reduce) {{
        * {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    
    /* Keyboard navigation indicators */
    .keyboard-hint {{
        background-color: color-mix(in srgb, #4299E1 15%, transparent);
        border: 2px solid #4299E1;
        border-radius: 8px;
        padding: 10px 15px;
        font-size: {font_size * 0.85}px;
        margin: 10px 0;
        display: inline-block;
    }}
    
    /* Screen reader only text */
    .sr-only {{
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0,0,0,0);
        white-space: nowrap;
        border-width: 0;
    }}
    
    /* Confirmation dialog */
    .confirmation-dialog {{
        background-color: color-mix(in srgb, #FFA500 20%, transparent);
        border: 4px solid #FFA500;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND RESOURCES
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    try:
        model = joblib.load('best_diabetes_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        features = pd.read_csv('model_features.csv')['features'].tolist()
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def show_progress_bar(current_step, total_steps=5):
    """Display a progress bar showing current step."""
    progress = (current_step / total_steps) * 100
    st.markdown(f"""
    <div class="step-indicator" role="status" aria-live="polite">
        <span class="sr-only">Progress: Step {current_step} of {total_steps}</span>
        📍 {t('step')} {current_step} {t('of')} {total_steps}
    </div>
    <div class="progress-bar" role="progressbar" aria-valuenow="{progress}" aria-valuemin="0" aria-valuemax="100" aria-label="Assessment progress">
        <div class="progress-fill" style="width: {progress}%;"></div>
    </div>
    """, unsafe_allow_html=True)

def show_help_button(help_text, key):
    """Display a help button with tooltip."""
    if st.button(f"❓ {t('need_help')}", key=f"help_{key}", help="Press H for help"):
        st.session_state.show_help[key] = not st.session_state.show_help.get(key, False)
    
    if st.session_state.show_help.get(key, False):
        st.markdown(f"""
        <div class="help-box" role="complementary" aria-label="Help information">
            <strong>ℹ️ {t('help_info')}</strong><br><br>
            {help_text}
        </div>
        """, unsafe_allow_html=True)

def show_keyboard_shortcuts():
    """Display keyboard shortcuts guide."""
    if st.session_state.show_keyboard_shortcuts:
        lang = st.session_state.language
        shortcuts_en = """
        ⌨️ **Keyboard Shortcuts:**
        - **Alt + N** or **→**: Next step
        - **Alt + B** or **←**: Previous step  
        - **Alt + H**: Toggle help
        - **Alt + R**: Restart assessment
        - **Tab**: Navigate between fields
        - **Space/Enter**: Select options
        """
        shortcuts_ms = """
        ⌨️ **Pintasan Papan Kekunci:**
        - **Alt + N** atau **→**: Langkah seterusnya
        - **Alt + B** atau **←**: Langkah sebelumnya
        - **Alt + H**: Togol bantuan
        - **Alt + R**: Mula semula penilaian
        - **Tab**: Navigasi antara medan
        - **Space/Enter**: Pilih pilihan
        """
        st.info(shortcuts_en if lang == 'en' else shortcuts_ms)

def inject_keyboard_handler():
    """Inject JavaScript to handle keyboard shortcuts."""
    import streamlit.components.v1 as components
    
    keyboard_js = """
    <script>
    (function() {
        // Remove existing listener if it exists
        if (window.keyboardHandlerAttached) {
            document.removeEventListener('keydown', window.handleKeyboard);
        }
        
        function findAndClickButton(textPatterns) {
            const buttons = parent.document.querySelectorAll('button');
            for (let btn of buttons) {
                const text = btn.innerText.toUpperCase();
                for (let pattern of textPatterns) {
                    if (text.includes(pattern.toUpperCase())) {
                        btn.click();
                        btn.focus();
                        return true;
                    }
                }
            }
            return false;
        }
        
        window.handleKeyboard = function(e) {
            // Only trigger with Alt key combinations (not arrow keys alone to avoid conflicts)
            if (!e.altKey) return;
            
            // Alt + N: Next
            if (e.altKey && e.key.toLowerCase() === 'n') {
                e.preventDefault();
                e.stopPropagation();
                findAndClickButton(['NEXT', 'SETERUSNYA', 'CALCULATE', 'KIRA']);
            }
            // Alt + B: Back
            else if (e.altKey && e.key.toLowerCase() === 'b') {
                e.preventDefault();
                e.stopPropagation();
                findAndClickButton(['BACK', 'KEMBALI']);
            }
            // Alt + H: Toggle Help
            else if (e.altKey && e.key.toLowerCase() === 'h') {
                e.preventDefault();
                e.stopPropagation();
                findAndClickButton(['Need Help', 'Perlukan Bantuan']);
            }
            // Alt + R: Restart
            else if (e.altKey && e.key.toLowerCase() === 'r') {
                e.preventDefault();
                e.stopPropagation();
                findAndClickButton(['START OVER', 'MULA SEMULA']);
            }
        };
        
        // Attach to parent document (Streamlit's main document)
        parent.document.addEventListener('keydown', window.handleKeyboard);
        window.keyboardHandlerAttached = true;
    })();
    </script>
    """
    
    components.html(keyboard_js, height=0)

def confirm_action(action_key, title, message, on_confirm):
    """Show confirmation dialog for critical actions."""
    st.markdown(f"""
    <div class="confirmation-dialog" role="alertdialog" aria-labelledby="confirm-title" aria-describedby="confirm-desc">
        <h3 id="confirm-title" style="margin-top: 0;">⚠️ {title}</h3>
        <p id="confirm-desc" style="font-size: {st.session_state.font_size}px;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    confirmed = False
    with col1:
        if st.button(f"✅ {t('yes')}", key=f"{action_key}_yes", use_container_width=True):
            confirmed = True
            on_confirm()
    with col2:
        if st.button(f"❌ {t('no')}", key=f"{action_key}_no", use_container_width=True):
            st.session_state.show_confirmation = False
            st.rerun()
    return confirmed

def get_probability_level(probability):
    """Categorize likelihood level based on probability."""
    if probability < 0.3:
        return "LOW", "prob-low", "🟢"
    elif probability < 0.6:
        return "MODERATE", "prob-medium", "🟡"
    else:
        return "HIGH", "prob-high", "🔴"

def generate_explanation(shap_values, feature_names, feature_values):
    """Generate human-readable explanations from SHAP values."""
    explanations = []
    
    # Get top contributing factors
    importance = list(zip(feature_names, shap_values, feature_values))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Bilingual explanation templates: (English name, English desc, Malay name, Malay desc)
    explanation_templates = {
        'GEN_HLTH': ("General Health", "Your overall health rating", "Kesihatan Am", "Penilaian kesihatan keseluruhan anda"),
        'BMI': ("Body Mass Index", "Your body weight relative to height", "Indeks Jisim Badan", "Berat badan anda berbanding ketinggian"),
        'AGE_GROUP': ("Age Group", "Your age category", "Kumpulan Umur", "Kategori umur anda"),
        'AGE': ("Age", "Your age", "Umur", "Umur anda"),
        'WGHT (lbs)': ("Weight", "Your body weight in pounds", "Berat Badan", "Berat badan anda dalam paun"),
        'CHKP_STATUS': ("Checkup Status", "How recently you had a medical checkup", "Status Pemeriksaan", "Bila pemeriksaan perubatan terakhir anda"),
        'ALHL_STATUS': ("Alcohol Status", "Your alcohol consumption habits", "Status Alkohol", "Tabiat pengambilan alkohol anda"),
        'CHOL_MEDS': ("Cholesterol Medication", "Whether you take cholesterol medication", "Ubat Kolesterol", "Sama ada anda mengambil ubat kolesterol"),
        'DCTR_STATUS': ("Doctor Visits", "Your frequency of doctor visits", "Lawatan Doktor", "Kekerapan lawatan doktor anda"),
        'EDUCATION_LEVEL': ("Education Level", "Your education level", "Tahap Pendidikan", "Tahap pendidikan anda"),
        'EMPLOYMENT_STATUS': ("Employment Status", "Your current employment status", "Status Pekerjaan", "Status pekerjaan semasa anda"),
        'SEX': ("Sex", "Your biological sex", "Jantina", "Jantina biologi anda"),
        'EXER_STATUS': ("Exercise Status", "Your physical activity level", "Status Senaman", "Tahap aktiviti fizikal anda"),
        'BMI_CATEGORY': ("BMI Category", "Your BMI classification", "Kategori BMI", "Klasifikasi BMI anda"),
        'BP_MEDS': ("Blood Pressure Medication", "Whether you take blood pressure medication", "Ubat Tekanan Darah", "Sama ada anda mengambil ubat tekanan darah"),
    }
    
    for feature, shap_val, feat_val in importance[:5]:
        if feature in explanation_templates:
            name_en, desc_en, name_ms, desc_ms = explanation_templates[feature]
        else:
            name_en = feature.replace('_', ' ').title()
            desc_en = f"Your {name_en.lower()}"
            name_ms = name_en
            desc_ms = desc_en
        
        direction = "increases" if shap_val > 0 else "decreases"
        impact = "significantly" if abs(shap_val) > 0.1 else "slightly"
        
        explanations.append({
            'feature_en': name_en,
            'feature_ms': name_ms,
            'description_en': desc_en,
            'description_ms': desc_ms,
            'direction': direction,
            'impact': impact,
            'shap_value': shap_val,
            'is_positive': shap_val > 0
        })
    
    return explanations

# ============================================================================
# MAIN APPLICATION - STEP-BY-STEP INTERFACE
# ============================================================================
def main():
    # Load model
    model, scaler, features = load_model()
    
    if model is None:
        st.error("⚠️ Could not load the prediction model. Please ensure the model files exist.")
        st.info("Required files: best_diabetes_model.pkl, feature_scaler.pkl, model_features.csv")
        return
    
    # ========================================================================
    # HEADER WITH ACCESSIBILITY CONTROLS
    # ========================================================================
    
    # Accessibility controls row
    col_font, col_lang, col_shortcuts = st.columns([2, 1.5, 1.5])
    
    with col_font:
        font_size_new = st.slider(
            "🔤 Text Size / Saiz Teks",
            min_value=16,
            max_value=28,
            value=st.session_state.font_size,
            step=2,
            key="font_size_slider",
            help="Adjust text size for better readability / Laraskan saiz teks untuk kebolehbacaan yang lebih baik"
        )
        if font_size_new != st.session_state.font_size:
            st.session_state.font_size = font_size_new
            st.rerun()
    
    with col_lang:
        lang_icon = "🇲🇾" if st.session_state.language == 'en' else "🇬🇧"
        lang_text = "BM" if st.session_state.language == 'en' else "EN"
        if st.button(f"{lang_icon} {lang_text}", key="lang_toggle", use_container_width=True, help="Switch language / Tukar bahasa"):
            st.session_state.language = 'ms' if st.session_state.language == 'en' else 'en'
            st.rerun()
    
    with col_shortcuts:
        if st.button("⌨️ Keys", key="show_shortcuts", use_container_width=True, help="Show keyboard shortcuts / Tunjuk pintasan papan kekunci"):
            st.session_state.show_keyboard_shortcuts = not st.session_state.show_keyboard_shortcuts
            st.rerun()
    
    # Keyboard shortcuts guide
    show_keyboard_shortcuts()
    
    # Inject keyboard event handler
    inject_keyboard_handler()
    
    # Language-specific title
    title = '🩺 Diabetes Probability Assessment Tool' if st.session_state.language == 'en' else '🩺 Alat Penilaian Kebarangkalian Diabetes'
    st.markdown(f'<h1 class="main-title" role="heading" aria-level="1">{title}</h1>', unsafe_allow_html=True)
    
    # Screen reader announcement for current section
    step_names = [
        "Basic Information / Maklumat Asas",
        "Physical Health / Kesihatan Fizikal", 
        "Health Conditions / Keadaan Kesihatan",
        "Lifestyle Habits / Tabiat Gaya Hidup",
        "Results / Keputusan"
    ]
    if st.session_state.current_step <= 5:
        st.markdown(f'<div class="sr-only" role="status" aria-live="polite">Current section: {step_names[st.session_state.current_step-1]}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box" role="region" aria-label="Welcome message">
        <strong style="font-size: {st.session_state.font_size * 1.2}px;">{t('welcome_title')}</strong><br><br>
        <span style="font-size: {st.session_state.font_size}px;">
        {t('welcome_msg')}<br><br>
        ✅ {t('easy_steps')}<br>
        ✅ {t('clear_results')}<br>
        ✅ {t('understand_factors')}<br>
        ✅ {t('get_recommendations')}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Keyboard navigation hint
    st.markdown(f"""
    <div class="keyboard-hint" role="note">
        💡 <strong>Tip:</strong> Use <kbd>Tab</kbd> to navigate, <kbd>Alt+N</kbd> for Next, <kbd>Alt+B</kbd> for Back
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # STEP-BY-STEP NAVIGATION
    # ========================================================================
    
    # Show current step progress
    show_progress_bar(st.session_state.current_step, total_steps=5)
    
    # STEP 1: Basic Information
    if st.session_state.current_step == 1:
        st.markdown(f'<h2 class="section-header" role="heading" aria-level="2">{t("step")} 1️⃣: {t("step1_title")}</h2>', unsafe_allow_html=True)
        
        show_help_button(t('step1_help'), "step1")
        
        # Screen reader context
        st.markdown('<span class="sr-only">Step 1 of 5: Please provide your basic demographic information. All fields are required.</span>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_group = st.selectbox(
                f"🎂 {t('age_label')}",
                options=list(range(1, 14)),
                format_func=lambda x: AGE_GROUPS[st.session_state.language][x],
                index=8,
                key="age_group"
            )
            
            sex = st.radio(
                f"⚧ {t('sex_label')}",
                options=[0, 1],
                format_func=lambda x: t('female') if x == 0 else t('male'),
                horizontal=True,
                key="sex"
            )
        
        with col2:
            education = st.selectbox(
                f"🎓 {t('education_label')}",
                options=[1, 2, 3, 4, 5, 6],
                format_func=lambda x: EDUCATION_LEVELS[st.session_state.language][x],
                index=3,
                key="education"
            )
            
            employment = st.selectbox(
                f"💼 {t('employment_label')}",
                options=[1, 2, 3, 4, 5, 6],
                format_func=lambda x: EMPLOYMENT_STATUS[st.session_state.language][x],
                index=3,
                key="employment"
            )
        
        st.session_state.user_data.update({
            'AGE_GROUP': age_group,
            'AGE': age_group,  # Using same value for both
            'SEX': sex,
            'EDUCATION_LEVEL': education,
            'EMPLOYMENT_STATUS': employment
        })
        
        if st.button(f"➡️ {t('next')}: {t('step2_title')}", use_container_width=True, help="Press Alt+N or → to continue"):
            st.session_state.current_step = 2
            st.rerun()
    
    # STEP 2: Physical Measurements
    elif st.session_state.current_step == 2:
        st.markdown(f'<h2 class="section-header" role="heading" aria-level="2">{t("step")} 2️⃣: {t("step2_title")}</h2>', unsafe_allow_html=True)
        
        # Screen reader context
        st.markdown('<span class="sr-only">Step 2 of 5: Please provide your physical health measurements including weight, height, and general health status.</span>', unsafe_allow_html=True)
        
        show_help_button(t('step2_help'), "step2")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight_kg = st.number_input(
                f"⚖️ {t('weight_label')}",
                min_value=20.0,
                max_value=300.0,
                value=st.session_state.user_data.get('weight_kg', 75.0),
                step=0.5,
                key="weight_kg"
            )
            
            height_cm = st.number_input(
                f"📏 {t('height_label')}",
                min_value=100.0,
                max_value=250.0,
                value=st.session_state.user_data.get('height_cm', 170.0),
                step=0.5,
                key="height_cm"
            )
            
            # Calculate BMI from weight (kg) and height (cm)
            height_m = height_cm / 100.0  # Convert cm to meters
            bmi = weight_kg / (height_m ** 2)
            
            # Display calculated BMI
            st.markdown(f"""
            <div class="info-box" style="padding: 15px; margin: 10px 0;">
                <strong>📊 {t('bmi_calculated')}:</strong> <span style="font-size: 24px; font-weight: bold;">{bmi:.1f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate BMI category
            if bmi < 18.5:
                bmi_category = 1  # Underweight
            elif bmi < 25:
                bmi_category = 2  # Normal
            elif bmi < 30:
                bmi_category = 3  # Overweight
            else:
                bmi_category = 4  # Obese
            
            # Convert weight from kg to pounds for the model (1 kg = 2.20462 lbs)
            weight_lbs = weight_kg * 2.20462
        
        with col2:
            gen_health = st.selectbox(
                f"❤️ {t('gen_health_label')}",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: HEALTH_RATING[st.session_state.language][x],
                index=st.session_state.user_data.get('GEN_HLTH', 3) - 1,
                key="gen_health"
            )
            
            checkup = st.selectbox(
                f"🩺 {t('checkup_label')}",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: CHECKUP_STATUS[st.session_state.language][x],
                index=0,
                key="checkup"
            )
        
        st.session_state.user_data.update({
            'weight_kg': weight_kg,
            'height_cm': height_cm,
            'WGHT (lbs)': weight_lbs,
            'BMI': bmi,
            'BMI_CATEGORY': bmi_category,
            'GEN_HLTH': gen_health,
            'CHKP_STATUS': checkup
        })
        
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button(f"⬅️ {t('back')}", use_container_width=True, help="Press Alt+B or ← to go back"):
                st.session_state.current_step = 1
                st.rerun()
        with col_next:
            if st.button(f"➡️ {t('next')}: {t('step3_title')}", use_container_width=True, help="Press Alt+N or → to continue"):
                st.session_state.current_step = 3
                st.rerun()
    
    # STEP 3: Health Conditions & Medications
    elif st.session_state.current_step == 3:
        st.markdown(f'<h2 class="section-header" role="heading" aria-level="2">{t("step")} 3️⃣: {t("step3_title")}</h2>', unsafe_allow_html=True)
        
        # Screen reader context
        st.markdown('<span class="sr-only">Step 3 of 5: Please provide information about your current health conditions and medications.</span>', unsafe_allow_html=True)
        
        show_help_button(t('step3_help'), "step3")
        
        bp_meds = st.radio(
            f"💊 {t('bp_meds_label')}",
            options=[0, 1],
            format_func=lambda x: f"✅ {t('yes_bp')}" if x == 1 else f"❌ {t('no_bp')}",
            index=st.session_state.user_data.get('BP_MEDS', 0),
            key="bp_meds"
        )
        
        chol_meds = st.radio(
            f"💊 {t('chol_meds_label')}",
            options=[0, 1],
            format_func=lambda x: f"✅ {t('yes_chol')}" if x == 1 else f"❌ {t('no_chol')}",
            index=st.session_state.user_data.get('CHOL_MEDS', 0),
            key="chol_meds"
        )
        
        doctor_visits = st.selectbox(
            f"👨‍⚕️ {t('doctor_visits_label')}",
            options=[1, 2, 3, 4],
            format_func=lambda x: DOCTOR_VISITS[st.session_state.language][x],
            index=st.session_state.user_data.get('DCTR_STATUS', 2) - 1,
            key="doctor_visits"
        )
        
        st.session_state.user_data.update({
            'BP_MEDS': bp_meds,
            'CHOL_MEDS': chol_meds,
            'DCTR_STATUS': doctor_visits
        })
        
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button(f"⬅️ {t('back')}", use_container_width=True, help="Press Alt+B or ← to go back"):
                st.session_state.current_step = 2
                st.rerun()
        with col_next:
            if st.button(f"➡️ {t('next')}: {t('step4_title')}", use_container_width=True, help="Press Alt+N or → to continue"):
                st.session_state.current_step = 4
                st.rerun()
    
    # STEP 4: Lifestyle Habits
    elif st.session_state.current_step == 4:
        st.markdown(f'<h2 class="section-header" role="heading" aria-level="2">{t("step")} 4️⃣: {t("step4_title")}</h2>', unsafe_allow_html=True)
        
        # Screen reader context
        st.markdown('<span class="sr-only">Step 4 of 5: Please provide information about your lifestyle habits including exercise and alcohol consumption.</span>', unsafe_allow_html=True)
        
        show_help_button(t('step4_help'), "step4")
        
        exercise = st.radio(
            f"🏋️ {t('exercise_label')}",
            options=[0, 1],
            format_func=lambda x: f"✅ {t('yes_exercise')}" if x == 1 else f"❌ {t('no_exercise')}",
            index=st.session_state.user_data.get('EXER_STATUS', 1),
            key="exercise"
        )
        
        alcohol = st.radio(
            f"🍺 {t('alcohol_label')}",
            options=[1, 2, 3, 4],
            format_func=lambda x: ALCOHOL_STATUS[st.session_state.language][x],
            index=st.session_state.user_data.get('ALHL_STATUS', 1) - 1,
            key="alcohol"
        )
        
        st.session_state.user_data.update({
            'EXER_STATUS': exercise,
            'ALHL_STATUS': alcohol
        })
        
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button(f"⬅️ {t('back')}", use_container_width=True, help="Press Alt+B or ← to go back"):
                st.session_state.current_step = 3
                st.rerun()
        with col_next:
            if st.button(f"➡️ {t('calculate')}", use_container_width=True, help="Press Enter to calculate your results"):
                st.session_state.current_step = 5
                st.rerun()
    
    # STEP 5: Results
    elif st.session_state.current_step == 5:
        st.markdown(f'<h2 class="section-header" role="heading" aria-level="2">📊 {t("results_title")}</h2>', unsafe_allow_html=True)
        
        # Screen reader context
        st.markdown('<span class="sr-only">Step 5 of 5: Your personalized diabetes risk assessment results and recommendations.</span>', unsafe_allow_html=True)
        
        # Prepare input data
        input_data = st.session_state.user_data.copy()
        
        # Create DataFrame with all required features
        input_df = pd.DataFrame([input_data])
        
        # Add any missing features with default values
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training
        input_df = input_df[features]
        
        try:
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction with loading animation
            with st.spinner(f'🔬 {t("analyzing")}'):
                time.sleep(1)  # Brief pause for better UX
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
            
            # Get probability level - translate the level
            level_key = 'LOW' if probability < 0.3 else ('MODERATE' if probability < 0.6 else 'HIGH')
            prob_level = t(level_key)
            prob_class = "prob-low" if probability < 0.3 else ("prob-medium" if probability < 0.6 else "prob-high")
            prob_emoji = "🟢" if probability < 0.3 else ("🟡" if probability < 0.6 else "🔴")
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            
            # Main result box
            st.markdown(f"""
            <div class="{prob_class}" role="region" aria-label="Diabetes risk assessment result">
                <div class="prob-text">{prob_emoji} {t('diabetes_likelihood')}: {prob_level}</div>
                <div class="probability-text">
                    {t('probability_score')}: {probability*100:.1f}%
                </div>
                <p style="font-size: {st.session_state.font_size * 0.9}px; margin-top: 15px;">
                    {t('current_profile')} <strong>{t(level_key.lower())}</strong> {t('likelihood_of')}
                </p>
                <span class="sr-only">Your diabetes risk level is {prob_level} with a probability score of {probability*100:.1f} percent.</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation with larger, clearer text
            if probability < 0.3:
                st.success(f"""
                ### ✅ {t('low_title')}
                
                {t('low_msg')}
                
                **{t('low_keep')}**
                - ✓ {t('low_1')}
                - ✓ {t('low_2')}
                - ✓ {t('low_3')}
                - ✓ {t('low_4')}
                
                **{'Remember' if st.session_state.language == 'en' else 'Ingat'}:** {t('low_remember')}
                """)
            elif probability < 0.6:
                st.warning(f"""
                ### ⚠️ {t('moderate_title')}
                
                {t('moderate_msg')}
                
                **{t('moderate_what')}**
                - 📞 {t('moderate_1')}
                - 🥗 {t('moderate_2')}
                - 🏃 {t('moderate_3')}
                - 📊 {t('moderate_4')}
                - ⚖️ {t('moderate_5')}
                
                **{'Good news' if st.session_state.language == 'en' else 'Berita baik'}:** {t('moderate_good')}
                """)
            else:
                st.error(f"""
                ### 🚨 {t('high_title')}
                
                {t('high_msg')}
                
                **{t('high_steps')}**
                - 🏥 {t('high_1')}
                - 🩸 {t('high_2')}
                - 💬 {t('high_3')}
                - 🥗 {t('high_4')}
                - 🏃 {t('high_5')}
                - ⚖️ {t('high_6')}
                
                **{'Remember' if st.session_state.language == 'en' else 'Ingat'}:** {t('high_remember')}
                """)
            
            st.markdown("---")
            
            # ================================================================
            # SHAP EXPLANATIONS
            # ================================================================
            st.markdown(f'<h2 class="section-header">💡 {t("understanding_title")}</h2>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box">
                <strong style="font-size: 22px;">{t('what_influences')}</strong><br><br>
                <span style="font-size: 19px;">
                {t('influences_msg')}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Calculate SHAP values
                with st.spinner('🔍 Analyzing your risk factors...'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_scaled)
                    
                    # Handle multi-output SHAP values
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    
                    # Generate explanations
                    explanations = generate_explanation(
                        shap_values[0], 
                        features, 
                        input_scaled[0]
                    )
                
                st.markdown(f"### 🔍 {t('top5_factors')}")
                
                for i, exp in enumerate(explanations, 1):
                    # Get language-specific feature name and description
                    feature_name = exp['feature_ms'] if st.session_state.language == 'ms' else exp['feature_en']
                    feature_desc = exp['description_ms'] if st.session_state.language == 'ms' else exp['description_en']
                    
                    if exp['is_positive']:
                        icon = f"⬆️ {t('increases_prob')}"
                        color = "#C53030"
                        card_class = "factor-card-positive"
                    else:
                        icon = f"⬇️ {t('decreases_prob')}"
                        color = "#276749"
                        card_class = "factor-card-negative"
                    
                    impact_text = f"{t('impact')} {t(exp['impact'])} {t(exp['direction'])} {t('your_prob')}"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <strong style="font-size: 26px;">{i}. {feature_name}</strong><br><br>
                        <span style="font-size: 20px; line-height: 1.7;">
                        {feature_desc} <span style="color: {color}; font-weight: bold; font-size: 22px;">{icon}</span><br><br>
                        <em>{impact_text}</em>
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.info("💭 Detailed factor analysis is being processed...")
            
            st.markdown("---")
            
            # ================================================================
            # PERSONALIZED RECOMMENDATIONS
            # ================================================================
            st.markdown(f'<h2 class="section-header">📝 {t("action_plan_title")}</h2>', unsafe_allow_html=True)
            
            recommendations = []
            
            # Generate personalized recommendations
            if input_data.get('BMI', 0) >= 25:
                recommendations.append({
                    'icon': '⚖️',
                    'title': t('rec_weight_title'),
                    'text': t('rec_weight_text'),
                    'action': t('rec_weight_action')
                })
            
            if input_data.get('EXER_STATUS', 1) == 0:
                recommendations.append({
                    'icon': '🏃',
                    'title': t('rec_exercise_title'),
                    'text': t('rec_exercise_text'),
                    'action': t('rec_exercise_action')
                })
            
            if input_data.get('GEN_HLTH', 3) >= 4:
                recommendations.append({
                    'icon': '❤️',
                    'title': t('rec_health_title'),
                    'text': t('rec_health_text'),
                    'action': t('rec_health_action')
                })
            
            if input_data.get('BP_MEDS', 0) == 1 or input_data.get('CHOL_MEDS', 0) == 1:
                recommendations.append({
                    'icon': '💊',
                    'title': t('rec_meds_title'),
                    'text': t('rec_meds_text'),
                    'action': t('rec_meds_action')
                })
            
            if input_data.get('ALHL_STATUS', 1) >= 3:
                recommendations.append({
                    'icon': '🍺',
                    'title': t('rec_alcohol_title'),
                    'text': t('rec_alcohol_text'),
                    'action': t('rec_alcohol_action')
                })
            
            if not recommendations:
                recommendations.append({
                    'icon': '✅',
                    'title': t('rec_good_title'),
                    'text': t('rec_good_text'),
                    'action': t('rec_good_action')
                })
            
            for rec in recommendations:
                st.markdown(f"""
                <div class="action-card">
                    <strong style="font-size: 24px;">{rec['icon']} {rec['title']}</strong><br><br>
                    <span style="font-size: 19px; line-height: 1.7;">
                    {rec['text']}<br><br>
                    <span class="action-highlight"><strong>👉 {t('action_step')}</strong></span> {rec['action']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # ================================================================
            # NEXT STEPS
            # ================================================================
            st.markdown("---")
            st.markdown(f'<h2 class="section-header">🎯 {t("next_steps_title")}</h2>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box">
                <strong style="font-size: 24px;">📋 {t('recommended_steps')}</strong><br><br>
                <span style="font-size: 20px; line-height: 2;">
                1️⃣ <strong>{t('next_1')}</strong><br>
                2️⃣ <strong>{t('next_2')}</strong><br>
                3️⃣ <strong>{t('next_3')}</strong><br>
                4️⃣ <strong>{t('next_4')}</strong><br>
                5️⃣ <strong>{t('next_5')}</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"⚠️ An error occurred while calculating your results: {str(e)}")
            st.info("Please try again or contact support if the problem persists.")
        
        # Navigation buttons
        st.markdown("---")
        col_restart, col_print = st.columns(2)
        
        with col_restart:
            # Show confirmation dialog if requested
            if st.session_state.get('show_confirmation', False):
                confirm_action(
                    'restart',
                    t('confirm_restart_title'),
                    t('confirm_restart_msg'),
                    lambda: restart_assessment()
                )
            else:
                if st.button(f"🔄 {t('start_over')}", use_container_width=True, help="Press Alt+R to restart"):
                    st.session_state.show_confirmation = True
                    st.rerun()
        
        with col_print:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;" role="note">
                <p style="font-size: {st.session_state.font_size * 0.9}px;">
                💡 <strong>{t('print_tip')}</strong> {t('print_msg')}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Always show footer and disclaimer at the bottom
    show_footer_and_disclaimer()

def restart_assessment():
    """Reset all session state to restart assessment."""
    st.session_state.current_step = 1
    st.session_state.user_data = {}
    st.session_state.show_confirmation = False
    st.rerun()

def show_footer_and_disclaimer():
    """Display the medical disclaimer and footer."""
    # ========================================================================
    # DISCLAIMER & FOOTER
    # ========================================================================
    st.markdown("---")
    
    # Medical Disclaimer Box
    font_size = st.session_state.get('font_size', 20)
    st.markdown(f"""
    <div style="background-color: color-mix(in srgb, #ff5555 10%, transparent); border: 4px solid #C53030; border-radius: 15px; padding: 30px; margin-top: 40px;" role="alert" aria-label="Medical disclaimer">
        <p style="color: #C53030; font-size: {font_size * 1.2}px; font-weight: bold; margin: 0 0 20px 0;">⚠️ {t('medical_disclaimer_title')}</p>
        <p style="font-size: {font_size * 0.95}px; line-height: 1.8; margin: 0 0 15px 0;">
        {t('disclaimer_msg')}
        </p>
        <ul style="font-size: {font_size * 0.95}px; line-height: 1.8; margin: 0 0 15px 20px;">
            <li>{t('disclaimer_1')}</li>
            <li>{t('disclaimer_2')}</li>
            <li>{t('disclaimer_3')}</li>
            <li>{t('disclaimer_4')}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <strong style="font-size: 18px;">Diabetes Probability Assessment Tool for Older Adults</strong><br>
        Developed for Research Project CSP760 (RO3)<br>
        Using CDC BRFSS 2023-2024 Data with Explainable AI (SHAP/LIME)<br>
        <br>
        <em>For Educational and Research Purposes Only</em><br>
        © 2026 - Not for Clinical Use
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

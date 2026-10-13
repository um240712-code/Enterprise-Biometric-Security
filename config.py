import os

# المسار الأساسي للمشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# مجلد حفظ صور المتسللين ومحاولات الاختراق
INTRUDERS_DIR = os.path.join(BASE_DIR, 'intruders')

# ملف تسجيل الأحداث الأمنية
LOG_FILE = os.path.join(BASE_DIR, 'security.log')
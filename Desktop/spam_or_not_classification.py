# -*- coding: utf-8 -*-
"""spam_or_not_classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1knoiB3hmqZSfKBCVXDzppDWNqkCUFVML
"""

from google.colab import drive
drive.mount('/content/drive',  force_remount=True)

!apt-get install -y git

!git config --global credential.helper 'cache --timeout=3600'

!git clone https://github.com/שם_משתמש/שם_הפרויקט.git

!git add .

!git config --global user.email "your.email@example.com"
!git config --global user.name "Your Name"

!git commit -m "תיאור של השינויים שביצעת"

!git push origin master

#ספריות
import numpy as np#מאפשר עיבוד מהיר של מערכים מרובי ממדים ופעולות מתמטיות עליהם
import pandas as pd#משמש לניתוח ועיבוד נתונים בטבלאות, מאגרי נתונים וקבצי CSV
import re#מספק כלים לעיבוד ופעולות על טקסט באמצעות פונקציות רגולריות.
import tensorflow as tf# ספריית למידת מכונה נרחבת, תחת המודול tf.keras המאפשרת בניית רשתות עם שכבות שונות, אופטימיזציה ועוד.
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub#מאפשר לטעון מודלים כבר מאומנים מספרטים של TensorFlow Hub כדי להשתמש בהם בקלות במודלים הנוכחיים.
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer

#import tokenization

!ls drive/MyDrive

!find drive/MyDrive

!unzip drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/spam_or_not/archive.zip

import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# קריאת המידע
df = pd.read_csv('drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/spam_or_not/spam_ham_dataset.csv', names=[ 'num','label','text','label_num'])
df = shuffle(df)
print(df['label_num'].dtype)
df['label_num'] = pd.to_numeric(df['label_num'], errors='coerce')
print(df['label_num'].dtype)
df['label_num'] = df['label_num'].fillna(0).astype(int)
print(df['label_num'].dtype)


# יצירת הטבלה החדשה
#df_3cols = pd.DataFrame({'label': df['label'], 'text': df['text'], 'label_num': df['label']})
df_2cols = df[['label_num', 'text']]



text = df.text.values
labels = df.label_num.values
print(len(text))
print(df.label_num.value_counts())

import matplotlib.pyplot as plt

a = df.label_num.value_counts()
plt.bar(a.index, a.values)
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Value Counts Chart')
plt.show()


# הצגת הטבלה החדשה
print(df_2cols.head())

import matplotlib.pyplot as plt#ייבוא של מודולים
import pandas as pd#ייבוא של מודולים

# Sample DataFrame creation (replace this with your actual DataFrame)


# Add a new column 'text_length' containing the length of each text
df['text_length'] = df['text'].apply(len)#חישוב אורך של סנטנס ורשימה של המידע בעמודה חדשה בדאטה

# Create a histogram
plt.hist(df['text_length'], bins=20, color='blue', edgecolor='black')#יצירת היסטוגרמה

# Add labels and title,טכני לגמרי!
plt.xlabel('Text Length')#הוספת תוויות להיסטוגרמה
plt.ylabel('Frequency')#הוספת תוויות להיסטוגרמה
plt.title('Histogram of Text Length')#הוספת כותרת

# Show the plot
plt.show()#הצגה של התרשים על המוסך

from sklearn.model_selection import train_test_split#ייבוא פונקציה שתפקידה לחלק נתונים לקבוצות של אימון ובדיקה
#df_2cols:מאגר הנתונים שלנו
train, test = train_test_split(df_2cols, test_size=0.2)#חלוקה ל20 אחוז בדיקה ו80 לאימון
train, val = train_test_split(train, test_size=0.2)#חלוקה של הנתונים לאימון, כך ש20 אחוז לישמש לאימות ו80 לאימון
#בעצם 20 אחוז משמש לבדיקה, 16 לאימות ו64 לאימון עצמו

print(train[0:10])#  הדפסת 10 ערכים ראשונים מתוך האימון

!pip install datasets  #התקנת חבילה ש מידע דרך מנהל חבילות
#למטה נראה כי ההורדה של החבילה אכן עבדה

from datasets import Dataset, DatasetDict #ייבוא מחלקות
import pandas as pd#ייבוא מחלקות
#DataFrame זה מבנה טבלאי שמסדר נתונים,לוקחים מתוך ספירת פנדס
train, test, val = pd.DataFrame(train),pd.DataFrame(test),pd.DataFrame(val)#סידור הנתונים בטבלה
#יצירת מילון
ds_dict = {'train' : Dataset.from_pandas(train),
           'val' : Dataset.from_pandas(val),
           'test' : Dataset.from_pandas(test)}
           #זה בעצם סידור הנתונים של אימון, אימות ובדיקה
#DatasetDict מבנה שמיועד לארגון נתונים
ds = DatasetDict(ds_dict)#הכלת המיון שיצרנו
print(ds)
#dataset.train_test_split(test_size=0.1)

ds['train'][1]

ds = ds.remove_columns("__index_level_0__")# הסרת עמודה על אינדקסים
ds

!pip install transformers #התקנת טרנספורמרים

from transformers import BertTokenizer#ייבוא ספריות

# Load the BERT tokenizer.
#, רק הכנה ,שימוש במודל "ברט בייס אנקייס" ובפונקציה ההופכת את כל התווים לאותיות קטנות, יקל בעתיד טוקיניזציה
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

from transformers import AutoTokenizer, DataCollatorWithPadding#ייבוא מחלקות וספריות

#הגדרת פונקציה
def tokenize_function(example):#ביצוע טוקינצזיה על החלק של "סנטנס"
    return tokenizer(example["text"], truncation=True)#בעצם משתמשים בטוקנייזר שהטענו מקודם, בעצם צריך לבנות מה אני רוצה שיקרה
#truncation: במידה והטקסט המוכנס לטוקניזר ארוך מדי, טוקניזר יחתוך אותו כך שיהיה באורך המוגדר

#ביצוע בפועל, לא צריך לשלוח example זה קורה אוטומטית
tokenized_datasets = ds.map(tokenize_function, batched=True)#יצירת קבוצת נתונים חדשה,, וביצוע טוקיניזציה על כל הקבוצה
#batched: כאשר משתמשים במאפ אז זה מסייע ליעילות, שזה יבצע על קבוצה ולא על כל אחד בנפרד, עוזר כשיש הרבה נתונים


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")#הכנה לאופוטימיזציה, ריפוד שיהיה באורך שווה,

print(df['label_num'].dtype)  #רק בדיקה שהטיפוס הוא כמו שצריך
df['label_num'] = pd.to_numeric(df['label_num'], errors='coerce')
print(df['label_num'].dtype)
df['label_num'] = df['label_num'].fillna(0).astype(int)
print(df['label_num'].dtype)

#input_ids: זהו רשימת מספרים המייצגים את המילים בטקסט לפי האינדקסים שלהם במילון המילים של המודל. כל מילה מומרת למספר, והמשפט עצמו מהווה רשימה של מספרים. ערך זה מכיל את המילים הממופות למספרים במילון המודל.

#attention_mask: זהו רשימת ביטים המציינת למודל אילו איברים יש להתעסק איתם ולאילו לא. בדרך כלל, הערך 1 מציין שמדובר במילה אמיתית והערך 0 מציין תא ריק או פדינג.

#token_type_ids: זהו מערך המציין למודל אילו משפט המילים השונים בקלט שיים לקטגוריות שונות. זה נמצא בשימוש במקרים שבהם יש שני קלטים טקסטואליים שיש להבדיל ביניהם, לדוגמה, במשימת שאלות ותשובות כאשר ישנה הבחנה בין שאלה לתשובה.
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["label_num"],#תוויות
    shuffle=True,#ערבוב נתונים, מעיד על הבנה של הרשת
    collate_fn=data_collator,#שיטת איסוף נתונים
    batch_size=4,#גודל קבוצה
)

#"token_type_ids" לשים לב שרלוונטי כשעושים קלסיפיקציה, אבל גם בעוד מקרים, פשוט כאן זה חשוב ועוזר

tf_validation_dataset = tokenized_datasets["val"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["label_num"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=4,
)

tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["label_num"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=4,
)

# הכנת הנתונים ל-TensorFlow Dataset
def prepare_dataset(dataset):
    input_ids = tf.convert_to_tensor(dataset["input_ids"])
    attention_mask = tf.convert_to_tensor([x if x is not None else [1] * len(dataset["input_ids"][0]) for x in dataset["attention_mask"]])
    token_type_ids = tf.convert_to_tensor(dataset["token_type_ids"])
    labels = tf.convert_to_tensor(dataset["label_num"])
    return tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids,
                                                 "attention_mask": attention_mask,
                                                 "token_type_ids": token_type_ids},
                                                labels))

from transformers import TFAutoModelForSequenceClassification#ייבוא ממחלקה

model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
#num_labels=2 מציין שישנן שתי תוויות אפשריות לסיווג המחרוזות במודל הזה (במקרה הזה, מדובר בבעיה של סיווג דו-קטגורי). לעומת זאת, במקרה של בעיה עם מספר תוויות שונות יש להתאים את המספר לכמות התוויות האפשרית.

# dont run this cell this is for check

from tensorflow.keras.losses import SparseCategoricalCrossentropy#ייבוא

model.compile(#קומפיילר
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
#"SparseCategoricalCrossentropy" היא פונקצית אובדן שמשמשת כפונקציית אובדן בשלב האימון של רשתות נוירונים, בדרך כלל בבעיות של סיווג מרובה קטגוריות כאשר הקטגוריות מיוצגות באופן ספרותי כמו תוויות.
model.fit(   #אימון
    tf_train_dataset,
    validation_data=tf_validation_dataset,
)

from tensorflow.keras.optimizers.schedules import PolynomialDecay#ייבוא פולינום עבור קצב למידה

batch_size =15 #גודל כל נגלה
num_epochs = 5 #מספר סיבובים
# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_train_steps = len(tf_train_dataset) * num_epochs #מספר שלבים כולל
lr_scheduler = PolynomialDecay(#קצב למידה
    initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)
from tensorflow.keras.optimizers import Adam#הגדרת האופטימייזר

opt = Adam(learning_rate=lr_scheduler)#עדכון של האופטימייזר לפי קצב למידה

#הגדרת הקומפיילר והאימון
import tensorflow as tf
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=5)

#חיזוי של התוצאה
import numpy as np
preds = model.predict(tf_test_dataset)["logits"]# מבצע חיזויים על קבוצת הבדיקה
class_preds = np.argmax(preds, axis=1) #השוואת החיזויים
print(preds.shape, class_preds.shape)# מדפיס את גודל המערך של הפלטים המקוריים מהמודל (שהם התוצאות של "logits") ואת גודל המערך של התוויות המתוחזקות (הקלאסים המוחזרים) שנוצרו מהפלט. מדובר כנראה במערך של תוצאות החיזויים ובמערך של התוויות שמצויות בכל דוגמה בקבוצת הבדיקה

#class_preds

from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict #ייבוא מחלקות
import pandas as pd

print(accuracy_score(test.label_num, class_preds))
print(f1_score(test.label_num, class_preds))
'''
הפונקציות accuracy_score ו־f1_score מתוך ספריית Scikit-Learn משמשות להערכת ביצועי המודל לאחר החיזויים ביחס לתוויות האמיתיות של קבוצת הבדיקה.

accuracy_score(test.label, class_preds): מחזירה את הדיוק של המודל על קבוצת הבדיקה. הפונקציה משווה בין תוויות האמת שבקבוצת הבדיקה (test.label) לבין התוויות שחזה המודל (class_preds), ומחזירה את הדיוק כמנת נכונות הזוגות שהתאימו זה לזה.

f1_score(test.label, class_preds): מחזירה את הפיתוח האחוד של המודל על קבוצת הבדיקה. הפיתוח האחוד הוא מדד הביצועים המשלב בין דיוק ורגישות (recall) של המודל. הפונקציה נותנת משקל שווה לדיוק ולרגישות, ומחשבה ממוצע ההפיתוח האחוד של המודל.

שתי הפונקציות עובדות על פי השוואה בין תוויות האמת לבין תוויות החיזוי של המודל ומחזירות מדדים שמעריכים את יכולת המודל לזהות נכון את התוויות של דוגמאות בקבוצת הבדיקה.'''
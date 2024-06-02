# Enhancing-Student-Summaries-A-Comparative-Study-of-DeBERTa-CNN-and-LightGBM

## Project Overview
So, disini project yang dibuat berguna untuk membangun model prediksi untuk skor konten dan "wording" dari ringkasan yang dibuat oleh siswa. Untuk dataset sendiri saya menggunakan open data dari kompetisi yang diselenggarakan di kaggle dengan kontributor utamanya adalah lembaga THE LEARNING AGENCY ->  [CommonLit Evaluate Student Summaries](https://www.kaggle.com/c/commonlit-evaluate-student-summaries)
 <br>
Data yang tersedia sebagai berikut.
- prompts_test.csv
- prompts_train.csv
- sample_submission.csv
- summaries_test.csv
- summaries_train.csv <br>
*Untuk pendekatan yang dilakukan pada project ini hanya dibatasi menggunakan data file summaries tanpa prompt.
## Data Overview
contoh sample data train yang digunakan sebagai berikut. <br>
| student_id    | prompt_id | text                                                                                                                                                                                                                                                                                                                                                                                                           | content          | wording          |
|---------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|------------------|
| 000e8c3c7ddb | 814d6b    | The third wave was an experimentto see how people reacted to a new one leader government. It gained popularity as people wanted to try new things. The students follow anything that is said and start turning on eachother to gain higher power. They had to stop the experement as too many people got to radical with it blindly following there leader,0.205682506482641,0.380537638762288 | 0.205682506482641 | 0.380537638762288 |
| 0020ae56ffbf | ebad26    | They would rub it up with soda to make the smell go away and it wouldnt be a bad smell. Some of the meat would be tossed on the floor where there was sawdust spit of the workers and they would make the meat all over again with the things in it.,-0.548304076980462,0.506755353548534                                                                                                               | -0.548304076980462 | 0.506755353548534 |
| 004e978e639e | 3b9047    | In Egypt, there were many occupations and social classes involved in day-to-day living. In many instances if you were at the bottom of the social ladder you could climb up, you didn't have to stay a peasant you could work to bring your status up. Everyone worshipped the gods Ra, Osiris, and Isis, but also they would worship their pharaohs like gods as well. Under the pharaohs were the priests, they had the responsibility to entertain or please the said god. The Chain of Command was placed to keep everyone in check, not one person could handle all the civilians and treasures without any aid. Like the tax collector, called a vizier like stated they were in charge of collecting the peoples' tax. They were also one of the rare instances who were able to read and write, that's how they were granted ""vizier"" Also the soldiers did many things as they would fight in wars or ""quelled domestic uprisings"". They were in charge of getting the slaves, farmers, and peasants to build palaces or the famous ancient pyramids. More skilled hardworking workers had occupations of craftsmen or women and physicians. This would mostly make up the middle-class people. The creative craftsmen would often make jewelry, papyrus products, pottery, tools, and many useful things people may need . Of course, you would need merchants to sell the goods to people who would pay for it. | 3.12892846350062 | 4.23122555224945 |
| 005ab0199905 | 3b9047    | The highest class was Pharaohs these people were gods.Then the 2nd highest class was a gonvener.Chiefs minister were called a vizier as a supervisor. (par.6),-0.210613934166593,-0.471414826967448                                                                                                                                                                                                              | -0.210613934166593 | -0.471414826967448 |
| 0070c9e7af47 | 814d6b    | The Third Wave developed  rapidly because the students genuinly believed that it was the best course of action. Their grades, acomplishments, and classparticipation/ behavior had improved dramatically since the experiment began. There did not seem to be any consiquenses in the students eyes. They became extremely engaged in all the Third Wave activites both inside and outside tha classroom. The experiment ended because the students were so patriotic about the ""movement"". The history class of thirty rapidly grew to 200 in three days.  That means 170 students joined a school ""movement"" in two days. Thats 85 people per day! On the fifth and final day all the students had completley believed that the ""Third Wave"" was a movement that would expell democracy. They believed a candidate from the ""movement"" would anounce its existance on television after five days of its success. The creater, Ron Jones, believed it had gone too far and for everyone's safety he shut it down.  If he hadn't the fake organization would have grown into something out of his controll. The Third Wave only lasted for a week. It could have spiralled into the American version of the Nazi Party, which is the opposite of what America stands for. | 3.27289414977436 | 3.21975651022738 |

<br> Data train yang digunakan berdimensi 7165 baris* 5 kolom. Nantinya data akan dilakukan splitting dengan skema holdout 80% untuk train dan 20% untuk data test.
## Modeling Approach
### 1st Tfidf+LightGBM
Untuk pendekatan pertama yang dilakukan disini saya hanya menggunakan tf-idf vectorizer sebagai feature extractor textnya lalu diteruskan ke lightgbm yang bertindak sebagai classifier. Flownya seperti pada gambar berikut. <br>
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1loEhBfGYHnXkwelQEmNBrGCuonyVH8YI" alt="mbconv">
</p> 

### 2nd Hybrid DeBERTa CNN  
Berikutnya untuk pendekatan yang kedua ini sedikit lebih kompleks. Gambaran kasarnya adalah menggunakan DeBERTa sebagai embedding dari input teksnya lalu diteruskan ke CNN block yang akan bertindak sebagai regressor yang memprediksi nilai output(content dan wording score), untuk arsitekurnya sebagai berikut.
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1UK8jEzj98vluCKEdNuJ7KMyRn3YWBPxa" alt="mbconv">
</p>

```
Untuk flownya mungkin dapat saya jelaskan secara singkat 
1. jadi input data dalam bentuk token yaitu input ids dan attention mask akan dimasukan
2. fitur teks akan diekstraksi dengan model DeBERTa(pretrained), model akan mengenerate output last hidden state dengan shape [batch size, sequence length, hidden size]
3. lakukan permute yang akan mengubah dimensi tensor yang semulanya [batch size, sequence length, hidden size] -> [batch_size, hidden_size, seq_len] (agar cocok dengan input yang diminta Conv1D)
4. masuk ke blok Conv1D pertama dengan konfigurasi input channels dari hidden size DeBERTa yang berjumlah 728 dan output channels yang dihasilkan adalah 128, untuk konfigurasi kernel, padding dll sudah ada didalam gambar
5. berikan fungsi aktivasi dan max pool
6. masuk ke blok Conv1D kedua dengan konfigurasi input channels 128 dan output channelsnya 64
7. sama seperti langkah sebelumnya, berikan fungsi aktivasi dan max pool
8. lakukan flattening agar shape dari tensor sesuai untuk dimasukkan ke layer fully connected
9. masuk ke fully connected pertama dan kedua
10. terakhir mapping hasil dari fully connected menjadi output sesuai dengan yang diinginkan
```

## Results
Hasil yang didapat dari 




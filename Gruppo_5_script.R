# LIBRERIE ----
# Dataset
library(readxl)
library(writexl)
# Directory
library(rstudioapi)
# Pre-processing
library(quanteda)
library(quanteda.textstats)
library(SnowballC) # stemming delle keywords
# Driver Analysis
library(dplyr)
library(syuzhet)
library(newsmap)
# Algoritmi
library(naivebayes)
library(randomForest)
library(iml)
library(future)
library(future.callr)
library(e1071)
library(reshape2)
library(cvTools)
library(caret)
# Grafici
library(ggplot2)
library(gridExtra)
library(quanteda.textplots)
library(kableExtra)
library(flextable)
library(officer)

# 1: DATASET E PULIZIA (Corpus, Dfm) ----

# Directory della cartella condivisa
setwd(dirname(getActiveDocumentContext()$path))
# Dataset
StoresReview <- read_excel("GRUPPO 3-4-5. Industry elettronica.xlsx")
# Aggiunta Primary key a sinistra del dataframe
StoresReview <- cbind(ID = seq(1:nrow(StoresReview)), StoresReview)

table(StoresReview$lang_value) # Presenza di altre lingue

# Eliminiamo le recensioni vuote e manteniamo solo quelle in lingua italiana.
Ita_StoresReview <- StoresReview[(StoresReview$lang_value == "it" |
                                    is.na(StoresReview$lang_value) == TRUE) & 
                                   is.na(StoresReview$text) == FALSE,]
# Putroppo l'algoritmo di deeplearning ha assegnato valori in lang_value 
# diversi da it e NA anche a delle recensioni in italiano, quindi li elimina.

table(Ita_StoresReview$social)
# Si nota che nel dataset, le recensioni provengono solo da twitter e places.

# Creazione corpus
Corpus_Totale <- corpus(Ita_StoresReview)

# Creazione dfm
Dfm_Totale <- dfm(tokens(Corpus_Totale,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_url = TRUE,
                         remove_numbers = TRUE) %>%
                    tokens_tolower() %>% 
                    tokens_remove(c(stopwords("italian"))) %>%
                    tokens_wordstem(language = "italian")) %>%
  dfm_trim(min_termfreq = 10,
           max_termfreq = 500, # Abbiamo messo un tetto per non considerare i 3 brand
           min_docfreq = 2)

# Raggruppiamo i tag e gli hashtag
Parole_Brutte <- colnames(Dfm_Totale)[grepl("^\\s*[#@]", trimws(colnames(Dfm_Totale)))] 
# Li togliamo
Dfm_Totale <- Dfm_Totale[,!(colnames(Dfm_Totale) %in% Parole_Brutte)]

# 2: ANALISI DEL SENTIMENT CON GLI ALGORITMI ----

# Campionamento con numerosità 200
# La qualità delle recensioni di places è superiore rispetto a quelle di twitter,
# Quindi abbiamo deciso di suddivere le due tipologie, prendendo 80% da places
# e il 20 % da twitter

# 2.1: PREPARAZIONE DEI DATI (training, test, matrici) ----
set.seed(001)
Training_places <- sample(Corpus_Totale[attr(Corpus_Totale, "docvars")$social == "places"],
                          size = 160,
                          replace = FALSE)
set.seed(002)
Training_tweet <- sample(Corpus_Totale[attr(Corpus_Totale, "docvars")$social == "twitter"],
                         size = 40,
                         replace = FALSE)

# Corpus per l'analisi manuale
Training_Corpus <- c(Training_tweet, Training_places)

# Corpus per il TEST SET
Test_Corpus <- Corpus_Totale[!(Corpus_Totale %in% Training_Corpus)]

# Verifica se sono complementari
setequal(Corpus_Totale, union(Test_Corpus, Training_Corpus))
# Risposta affermativa, ma si nota una differenza di 21 testi

# Dataset del Campione per poterlo esportare
Campione <- data.frame(
  ID = attr(Training_Corpus, "docvars")$ID,
  persona <- rep(c("William","Davide","Maddalena","Giacomo"),each = 50),
  Training_Corpus,
  sentiment <- NA)
names(Campione) <- c("ID","Persona","text","sentiment")
# Rinominazione necessaria per far riconoscere ad RStudio il text field

#Esportare il Campione
#write_xlsx(Campione, "Training Data Grezzo.xlsx") # NON RUNNARE !!!!!!!!!!!!!!!!!!!!!!!
Campione <- read_excel("Training Data Grezzo.xlsx")

Campione$sentiment <- ifelse(Campione$sentiment == -1, "Negativo", 
                             ifelse(Campione$sentiment == 0, "Neutro", 
                                    "Positivo"))

# Verifica celle vuote.
apply(Campione, 2, function(x) sum(is.na(x)))
# Ogni recensione ha una valutazione

# Conversione in corpus con la variabile del sentiment
Training_Corpus <- corpus(Campione)

Dfm_Training <- dfm(tokens(Training_Corpus,
                           remove_punct = TRUE,
                           remove_symbols = TRUE,
                           remove_url = TRUE,
                           remove_numbers = TRUE) %>%
                      tokens_tolower() %>% 
                      tokens_remove(c(stopwords("italian"))) %>%
                      tokens_wordstem(language = "italian")) %>%
  dfm_trim(max_termfreq = 500) # I Brand abbassavano l'accuracy

Dfm_Training <- Dfm_Training[,!(colnames(Dfm_Training) %in% Parole_Brutte)]


Dfm_Test <- dfm(tokens(Test_Corpus,
                       remove_punct = TRUE,
                       remove_symbols = TRUE,
                       remove_url = TRUE,
                       remove_numbers = TRUE) %>%
                  tokens_tolower() %>% 
                  tokens_remove(c(stopwords("italian"))) %>%
                  tokens_wordstem(language = "italian")) %>%
  dfm_trim(max_termfreq = 500)

Dfm_Test <- Dfm_Test[,!(colnames(Dfm_Test) %in% Parole_Brutte)]

length(Dfm_Training@Dimnames$features) #1235
length(Dfm_Test@Dimnames$features) #8533

# Abbasso il numero di features del test
Dfm_Test <- dfm_match(Dfm_Test, 
                      features = featnames(Dfm_Training))
# Dopo il match lunghezzze pari a 1235

# Verifica
setequal(featnames(Dfm_Training), 
         featnames(Dfm_Test))

# Creazione matrici per gli algoritmi
Matrice_Training <- as.matrix(Dfm_Training)
Matrice_Test <- as.matrix(Dfm_Test)

# Conversione del vettore sentiment in factor
Dfm_Training@docvars$sentiment <- as.factor(Dfm_Training@docvars$sentiment)

# 2.2: TRAINING E CLASSIFICAZIONE (training e prediction) ----

# ALGORITMO NAIVE BAYES
set.seed(123)
NaiveBayesModel <- multinomial_naive_bayes(x=Matrice_Training,
                                           y=Dfm_Training@docvars$sentiment,
                                           laplace = 1)
# Distribuzione del sentiment sul training
summary(NaiveBayesModel)

# Predizione sul test set
Test_predictedNB <- predict(NaiveBayesModel,
                            Matrice_Test)

# frequenze assolute sui 3400 testi del test
table(Test_predictedNB)
Test_predictedNB

# ALGORITMO RANDOM FOREST
set.seed(150)
RF <- randomForest(y= Dfm_Training@docvars$sentiment,
                   x= Matrice_Training,
                   importance=TRUE,
                   do.trace=FALSE,
                   ntree=500)
RF # Error rate: 31%
table(Campione$sentiment)

# Linee NON tratteggiate
  plot(RF, type = "l", col = c("black", "steelblue4", "violetred4", "springgreen4"), lty = 1,
       main = "Errori del modello Random Forest: variabile sentiment") +   
  legend("topright", horiz = FALSE, cex = 0.7, title = "Errori:",       
         fill = c("springgreen4", "black", "steelblue4", "violetred4"),        
         legend = c("Positivo", "Medio", "Negativo", "Neutro"))
  
Errori <- as.data.frame(RF$err.rate)

which.min(Errori$OOB) # 23

set.seed(150)
RF <- randomForest(y= Dfm_Training@docvars$sentiment,
                   x= Matrice_Training,
                   importance=TRUE,
                   do.trace=FALSE,
                   ntree=23)
RF # 28% error rate

Test_predictedRF <- predict(RF, Matrice_Test ,type="class")
table(Test_predictedRF)

# SUPPORT VECTOR
set.seed(175)
SupportVectorMachine <- svm(
  y= Dfm_Training@docvars$sentiment,
  x=Matrice_Training, kernel='linear', cost = 1)

length(SupportVectorMachine$index) # 173 support vector

Test_predictedSV <- predict(SupportVectorMachine, Matrice_Test)

table(Test_predictedSV)

Confronto_test <- data.frame(
  ID = attr(Test_Corpus, "docvars")$ID,
  social = attr(Test_Corpus, "docvars")$social,
  Test_Corpus,
  Test_predictedNB,
  Test_predictedRF,
  Test_predictedSV)

results <- as.data.frame(rbind(prop.table(table(Test_predictedNB)),
                               prop.table(table(Test_predictedRF)),
                               prop.table(table(Test_predictedSV))))

results$algorithm <- c("Naive Bayes", "Random Forest", "Support Vector Machine")

df.long<-melt(results,id.vars=c("algorithm"))

ggplot(df.long,aes(algorithm,value,fill=variable))+
  geom_bar(position="dodge",stat="identity") + scale_fill_manual(values = c("violetred3", "yellow3", "orange2")) +
  labs(title = "Comparazione delle predizioni") +
  theme(axis.text.x = element_text(color="#993333", angle=90)) + coord_flip() +
  ylab(label="Proporzione delle categorie nel test set") + xlab("Algoritmi") +
  guides(fill=guide_legend(title="Categorie di \nsentiment")) +
  theme(plot.title = element_text(color = "black", size = 12, face = "plain"),
        axis.title=element_text(size=11,face="plain"),
        axis.text= element_text(size =10, face = "italic")
  )

# 2.3: CROSS VALIDATION ----

Matrice_Training2 <- Matrice_Training

 
set.seed(200)
k <- 5
folds <- cvFolds(NROW(Matrice_Training2), K = k)

for(i in 1:k){
  Matrice_Training <-
    Matrice_Training2 [folds$subsets[folds$which != i], ]
  ValidationSet <-
    Matrice_Training2 [folds$subsets[folds$which == i], ]
  set.seed(200)
  NaiveBayesModel <- multinomial_naive_bayes(
    y= Dfm_Training[folds$subsets[folds$which != i], ]
    @docvars$sentiment ,
    x=Matrice_Training, laplace = 1)
  Predictions_NB <- predict(NaiveBayesModel, 
                            newdata = ValidationSet, 
                            type = "class")
  class_table <- table("Predictions"= Predictions_NB,
                       
                       "Actual"=Dfm_Training[folds$subsets[folds$which == i], ]@docvars$sentiment)
  
  print(class_table)
  df<-confusionMatrix( class_table, mode = "everything")
  df_measures_NB<-paste0("conf.mat.nb",i)
  assign(df_measures_NB,df)
}

NB_Prediction <- data.frame(col1=vector(), col2=vector(), col3=vector(), col4=vector())

#Riempiamo il dataset con i valori di accuracy e f1 
for(i in mget(ls(pattern = "conf.mat.nb")) ) {
  Accuracy <-(i)$overall[1]
  p <- as.data.frame((i)$byClass)
  F1_negative <- p$F1[1]
  F1_neutral <- p$F1[2]
  F1_positive <- p$F1[3]
  NB_Prediction <- rbind(NB_Prediction , cbind(Accuracy , F1_negative ,
                                               F1_neutral, F1_positive ))
  
}

str(NB_Prediction) # Si nota la presenza di NA
NB_Prediction [is.na(NB_Prediction )] <- 0

AverageAccuracy_NB <- mean(NB_Prediction[, 1] )
AverageF1_NB<- mean(colMeans(NB_Prediction[-1] ))
AverageAccuracy_NB # 81,5%
AverageF1_NB # 72,7%


# RANDOM FOREST
for(i in 1:k){
  Matrice_Training <-
    Matrice_Training2 [folds$subsets[folds$which != i], ]
  ValidationSet <-
    Matrice_Training2 [folds$subsets[folds$which == i], ]
  set.seed(250)
  RandomForest <- randomForest(
    y= Dfm_Training[folds$subsets[folds$which != i], ]
    @docvars$sentiment ,
    x=Matrice_Training, do.trace=FALSE, ntree=23)
  Predictions_RF <- predict(RandomForest, 
                            newdata= ValidationSet, 
                            type="class")
  class_table <- table("Predictions"= Predictions_RF,
                       "Actual"=Dfm_Training[folds$subsets[folds$which == i], ]@docvars$sentiment)
  print(class_table)
  df<-confusionMatrix( class_table, mode = "everything")
  df_measures_RF<-paste0("conf.mat.rf",i)
  assign(df_measures_RF,df)
}

RF_Predictions <- data.frame(col1=vector(), col2=vector(), col3=vector(), col4 = vector())


for(i in mget(ls(pattern = "conf.mat.rf")) ) {
  Accuracy <-(i)$overall[1]
  p <- as.data.frame((i)$byClass)
  F1_negative <- p$F1[1]
  F1_neutral <- p$F1[2]
  F1_positive <- p$F1[3]
  RF_Predictions <- rbind(RF_Predictions , cbind(Accuracy , F1_negative ,
                                                 F1_neutral, F1_positive ))
  
}

str(RF_Predictions) # Presenza NA
RF_Predictions [is.na(RF_Predictions )] <- 0


AverageAccuracy_RF <- mean(RF_Predictions[, 1] )
AverageF1_RF<- mean(colMeans(RF_Predictions[-1] ))

AverageAccuracy_RF # 67,5%
AverageF1_RF  #52,6%


# SUPPORT VECTOR MACHINE
for(i in 1:k){
  Matrice_Training <-
    Matrice_Training2 [folds$subsets[folds$which != i], ]
  ValidationSet <-
    Matrice_Training2 [folds$subsets[folds$which == i], ]
  set.seed(300)
  SupportVectorMachine<- svm(
    y= Dfm_Training[folds$subsets[folds$which != i], ]
    @docvars$sentiment, 
    x=Matrice_Training, kernel='linear', cost = 1)
  Prediction_SVM <- predict(SupportVectorMachine,
                            newdata=ValidationSet)
  class_table <- table("Predictions"= Prediction_SVM,
                       "Actual"=Dfm_Training[folds$subsets[folds$which == i], ]@docvars$sentiment)
  print(class_table)
  df<-confusionMatrix( class_table, mode = "everything")
  df_measures_SVM<-paste0("conf.mat.sv",i)
  assign(df_measures_SVM,df)
}


SVM_Prediction <- data.frame(col1=vector(), col2=vector(), col3=vector(), col4=vector())

#Riempiamo il dataframe 
for(i in mget(ls(pattern = "conf.mat.sv")) ) {
  Accuracy <-(i)$overall[1]
  p <- as.data.frame((i)$byClass)
  F1_negative <- p$F1[1]
  F1_neutral <- p$F1[2]
  F1_positive <- p$F1[3]
  SVM_Prediction <- rbind(SVM_Prediction , cbind(Accuracy , F1_negative ,
                                                 F1_neutral, F1_positive ))
  
}

str(SVM_Prediction) # Presenza NA
SVM_Prediction [is.na(SVM_Prediction)] <- 0


#Calcoliamo i valori medi
AverageAccuracy_SVM <- mean(SVM_Prediction[, 1] )
AverageF1_SVM<- mean(colMeans(SVM_Prediction[-1] ))

AverageAccuracy_SVM # 73%
AverageF1_SVM # 62,2%


# CONFRONTO
AccNB <- as.data.frame(AverageAccuracy_NB )
colnames(AccNB)[1] <- "NB"

#Creo un dataframe per RF
AccRF <- as.data.frame(AverageAccuracy_RF )
#Rinomino la colonna
colnames(AccRF)[1] <- "RF"

#Creo un dataframe per SVM
AccSVM<- as.data.frame(AverageAccuracy_SVM )
#Rinomino la colonna
colnames(AccSVM)[1] <- "SVM"

#Unisco in un unico dataframe i valori di accuracy dei tre modelli
Accuracy_models <- cbind(AccNB, AccRF, AccSVM)
Accuracy_models

Accuracy_models_Melt <-melt(Accuracy_models)

plot_accuracy <- ggplot(Accuracy_models_Melt, aes(x=variable, y=value, color = variable)) +
  geom_boxplot() + xlab("Algorithm") + ylab(label="Values of accuracy") +
  labs(title = "Cross-validation with k =5: values of accuracy") + coord_flip() +
  theme_bw() +
  guides(color=guide_legend(title="Algorithms")) +
  theme(plot.title = element_text(color = "black", size = 12, face = "italic"),
        axis.title.x =element_text(size=12,face="bold"),
        axis.title.y =element_text(size=12, face = "plain"),
        axis.text= element_text(size =10, face = "italic")
  )

F1NB <- as.data.frame(AverageF1_NB)
colnames(F1NB)[1] <- "NB"
#RF
F1RF<- as.data.frame(AverageF1_RF )
colnames(F1RF)[1] <- "RF"
#SVM
F1SVM <- as.data.frame(AverageF1_SVM)
colnames(F1SVM)[1] <- "SVM"
#DATAFRAME
f1_models <- cbind(F1NB, F1RF, F1SVM)
f1_models

f1_models_melt <-melt(f1_models)
str(f1_models_melt)

plot_f1 <- ggplot(f1_models_melt, aes(x=variable, y=value, color = variable)) +
  geom_boxplot() + xlab("Algorithm") + ylab(label="Values of f1") +
  labs(title = "Cross-validation with k =5: values of f1") + coord_flip() +
  theme_bw() +
  guides(color=guide_legend(title="Algorithms")) +
  theme(plot.title = element_text(color = "black", size = 12, face = "italic"),
        axis.title.x =element_text(size=12,face="bold"),
        axis.title.y =element_text(size=12, face = "plain"),
        axis.text= element_text(size =10, face = "italic")
  )

grid.arrange(plot_accuracy, plot_f1, nrow=2) #bayes
str(Test_Corpus)

Esiti_algo_Test <- data.frame(
  ID = attr(Test_Corpus, "docvars")$ID,
  social = attr(Test_Corpus, "docvars")$social,
  text = Test_Corpus,
  Bayes = Test_predictedNB,
  RF = Test_predictedRF,
  SVM = Test_predictedSV
)

Df_sentiment <- data.frame(
  ID = c(Training_Corpus$ID, Test_Corpus$ID),
  NB_sentiment = c(Campione$sentiment, as.vector(Test_predictedNB))
)

Ita_StoresReview <- merge(Ita_StoresReview, Df_sentiment, by='ID')

set.seed(007)
AAP_test_data_places <- Confronto_test %>%
  filter(social == 'places') %>%
  slice_sample(n=40, replace = FALSE)
# Con questo codice estraggo 40 random samples dal testing data con social == places

set.seed(007)
AAP_test_data_twitter <- Confronto_test %>%
  filter(social == 'twitter') %>%
  slice_sample(n=20, replace = FALSE)

# AAP_test_data <- rbind(AAP_test_data_places, AAP_test_data_twitter)
# Esporto i due dataframe. Perchè non li unisco? Perchè questo linguaggio è SPAZZATURA e da errori strani.
# Farò il bind manuale su excel
#write_xlsx(AAP_test_data_places, 'aap_sample_places.xlsx')
#write_xlsx(AAP_test_data_twitter, 'aap_sample_twitter.xlsx')

AAP_test_data <- read_excel("aap_reviewed.xlsx")
AAP_test_data$human_val <- ifelse(AAP_test_data$human_val  == -1, "Negativo", 
                                  ifelse(AAP_test_data$human_val  == 0, "Neutro", 
                                         "Positivo"))

AAP_confronto <- c(Bayes = sum(AAP_test_data$Test_predictedNB == AAP_test_data$human_val),
                   RF = sum(AAP_test_data$Test_predictedRF == AAP_test_data$human_val),
                   SV = sum(AAP_test_data$Test_predictedSV == AAP_test_data$human_val))

AAP_confronto <- data.frame(
  Bayes = c(sum(AAP_test_data$Test_predictedNB == AAP_test_data$human_val),
            sum(AAP_test_data$Test_predictedNB == AAP_test_data$human_val)/60*100),
  
  RF = c(sum(AAP_test_data$Test_predictedRF == AAP_test_data$human_val),
         sum(AAP_test_data$Test_predictedRF == AAP_test_data$human_val)/60*100),
  
  SV = c(sum(AAP_test_data$Test_predictedSV == AAP_test_data$human_val),
         sum(AAP_test_data$Test_predictedSV == AAP_test_data$human_val)/60*100)
)
rownames(AAP_confronto) <- c("Freq Assoluta", "Freq Relativa")

# 3: DRIVER ANALYSIS ----

# Frequenze delle caratteristiche del Corpus
apply(textstat_summary(Corpus_Totale)[,2:11], 2, sum)

  
Dfm_Places <- dfm(tokens(Corpus_Totale[attr(Corpus_Totale, "docvars")$social == "places"],
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_url = TRUE,
                         remove_numbers = TRUE) %>%
                    tokens_tolower() %>% 
                    tokens_remove(c(stopwords("italian"))) %>%
                    tokens_wordstem(language = "italian")) %>%
  dfm_trim(min_termfreq = 10,
           max_termfreq = 500, # Abbiamo messo un tetto per non considerare i 3 brand
           min_docfreq = 2)

# RILEVAZIONE DELLE KEYWORDS

# Top parole del DFM
topfeatures(Dfm_Places,50)

# DA RIVEDERE
Parole_Popolari <- textstat_frequency(Dfm_Places, n =50)
Parole_Popolari$feature <- with(Parole_Popolari, reorder(feature, frequency))

ggplot(Parole_Popolari, aes(x=frequency, y=feature)) +
 geom_point(size = 1.5, color = "Darkorange2") +
  theme_bw() +
theme(axis.text.x = element_text(angle=360, hjust=1)) +
labs(x = "Features", y = "Frequenza", 
      title = "Le 20 parole più frequenti nelle recensioni") +
  theme(plot.title = element_text(color = "Darkorange2", size = 11, face = "bold"),
        plot.subtitle = element_text(color = "black", size = 11, face = "italic" ))

textplot_wordcloud(Dfm_Places, 
                  min_size = 1.5,  
                  max_size = 4,    
                   min.count = 10,   
                   max_words = 50,  
                   random.order = FALSE,  
                   random_color = FALSE,
                   rotation = 0,    #rotazione delle parole
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))


# 3.1: KEYWORDS E DRIVER ----
Driver <- dictionary(list(prezzo = c("promozione", "risparmio", "qualità", "prezzo", "economicità", 
                                     "economico", "concorrenziali", "sconto", 
                                     "offerta", "budget", "ragionevole","costo", "sostenibile", 
                                     "convenienti", "sottocosto", "eccezional", "super", "miglior", "ben", 
                                     "top", "futur", "offert", "convenient", "risparm", "assurd", "super", "pazzesc", "gratis"),
                          servizio =  c("rapidità", "Empatia", "professionale", "supporto", 
                                        "risoluzione","problemi",  "cordialità", "assistenza", "vendita", 
                                        "immediata", "efficienza", "cortese", "reclami", 
                                        "competenza", "cliente", "flessibilità", "tempestività", 
                                        "servizio", "accoglienza", "caloroso", "gentile", "personal", "competente", "disponibile", "male",
                                        "lento", "disorganizzato", "disordinato", "scortese", "cafone", "garanzia", "reso", "account",
                                        "signor", "reparto", "richiest", "graz", "eccezional", "inform", "miglior", "ragazz",
                                        "rispost", "gent", "gentilissim", "rispett", "competent", "bell", "ringraz", "aiut",
                                        "pront", "addett", "pessim", "pazienz", "ore", "benissim", "purtropp", "purtropp", "problem",
                                        "incompetent", "rivolg", "compl", "ben", "consigl", "prossim", "buon", "gentilezz", "educ",
                                        "simpat", "dispon", "attenzion", "qualif", "aspett", "grandissim", "disponibil", "esigent", "top",
                                        "giovan", "assist", "futur", "risolt", "bravissim", "commess", "brav", "spieg", "dubb", "vergogn",
                                        "inutil", "maleduc", "pochissim", "signorin", "bellissim", "perfett", "attent", "super",
                                        "pazzesc", "soluzion", "difett", "truff", "qualit", "normal", "scortes", "intelligent"),
                          ordini = c("transazione", "acquisto", "pagamento", "tempo", "consegna", 
                                     "ordine", "opzioni", "modalità", "ritiro", 
                                     "rimborso", "conferma", "tracciabilità", "facilità", 
                                     "catalogo", "online","checkout", "garanzia", "reso", "account", "bell", "pessim", "nuov",
                                     "benissim", "purtropp", "problem", "ben", "buon", "aspett", "attesa", "top", "futur", "risolt",
                                     "vergogn", "inutil", "bellissim", "assurd", "super", "difett", "truff"),
                          location = c("accesso", "facilitato", "ampio", "parcheggio", "zona", "geografica", 
                                       "ambiente", "accogliente", "strutture", "moderne",
                                       "punto", "vendita", "facilità", "raggiungimento", "accessibilità", "disabili", 
                                       "prossimità","area", "centrale", "sicurezza", "atmosfera", "piacevole", "posizione","strategica", 
                                       "illuminata", "spazio", "facile", 
                                       "tranquilla", "negozio", "posto", "affollato", "piccolo", "disordinato", "bell", "ben")))

stem_words <- function(words) {
  stemmed_words <- wordStem(words, language = "italian")
  return(stemmed_words)
}

Driver$prezzo <- stem_words(Driver$prezzo)
Driver$servizio <- stem_words(Driver$servizio)
Driver$ordini <- stem_words(Driver$ordini)
Driver$location <- stem_words(Driver$location)

Driver_Review <- dfm_lookup(Dfm_Totale,Driver)

Driver_Conv_Rev <- convert(Driver_Review, to = "data.frame")
Driver_Conv_Rev <- cbind(ID = Dfm_Totale@docvars$ID, Driver_Conv_Rev)

apply(Driver_Conv_Rev[,3:6],2,sum)
prop.table(apply(Driver_Conv_Rev[,3:6],2,sum))


DriverAnalysis <- full_join(Ita_StoresReview, Driver_Conv_Rev)

# 3.2: SENTIMENT ANALYSIS ----

Dizionario <- get_sentiment_dictionary(dictionary = 'nrc', 
                                       language = "italian")

Store_reviews_sentiment <- get_sentiment(Corpus_Totale,
                                                      method = 'nrc', language = "italian")

DriverAnalysis$sentimentAnalysis <- Store_reviews_sentiment

DriverAnalysis$sentiment_labels <- ifelse(DriverAnalysis$sentimentAnalysis <= 0, "Negativo", "Positivo")

RatingXsentiment <- table(DriverAnalysis$sentiment_labels, DriverAnalysis$score_rating)
RatingXsentiment <- as.data.frame(RatingXsentiment)
colnames(RatingXsentiment) <- c("Sentiment","Rating","Freq")

ggplot(RatingXsentiment,aes(Rating, Freq, fill=Sentiment))+
  geom_bar(position="stack",stat="identity") +   
  scale_fill_manual(values = c("#CA3432", "darkseagreen")) +
  labs(title = "Come cambia il valore del sentiment al variare del rating?") +
  theme(axis.text.x = element_text(color="#993333", angle=90)) + 
  coord_flip() +
  ylab(label="Valori assoluti") + 
  xlab("Rating") +
  guides(fill=guide_legend(title="sentiment")) +
  theme(plot.title = element_text(color = "black", size = 12, face = "plain"),
        plot.subtitle = element_text(face = "plain"),
        axis.title=element_text(size=10,face="plain"),
        axis.text= element_text(size =10, face = "italic"))

# 3.3: EMOTION ANALYSIS ----

EmotionAnalysis <- get_nrc_sentiment(Corpus_Totale)

barplot(
  sort(colSums(prop.table(EmotionAnalysis[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Le emozioni nel corpus", xlab="Proporzioni",
  col = "#8C96C6"
)


# 3.4: NEWSMAP ----

DriverAnalysis_SemiSupervisedApproach <- dfm(tokens(Corpus_Totale,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_url = TRUE,
                         remove_numbers = TRUE) %>%
                           tokens_tolower() %>%
                           tokens_wordstem(language = "italian") %>%
                           tokens_lookup(dictionary = Driver))

TextModel <- textmodel_newsmap(Dfm_Totale,
                               DriverAnalysis_SemiSupervisedApproach)

DriverAnalysis$SemiSupervised <- predict(TextModel)
round(prop.table(table(predict(TextModel))), 2)*100

DriverAnalysis$Dizionario <- ifelse(DriverAnalysis$prezzo > 0, "Prezzo",
                                    ifelse(DriverAnalysis$servizio > 0, "Servizio",
                                           ifelse(DriverAnalysis$ordini > 0, "Ordini",
                                                  ifelse(DriverAnalysis$location > 0, "Location", "NA"))))

ConfrontoRisultati <- filter(DriverAnalysis, Dizionario == "NA")
DriverAnalysis$doc_id <- NULL

# 4 ----

# TABELLA GENERALE
flex_table_stores = data.frame(unclass(table(Ita_StoresReview$Player, Ita_StoresReview$NB_sentiment)))
flex_table_stores$media_rating = round(c(mean(Ita_StoresReview$score_rating[Ita_StoresReview$Player=='Euronics'],na.rm = TRUE),
                                         mean(Ita_StoresReview$score_rating[Ita_StoresReview$Player=='Mediaworld'],na.rm = TRUE),
                                         mean(Ita_StoresReview$score_rating[Ita_StoresReview$Player=='Unieuro'],na.rm = TRUE)), 2)
flex_table_stores = cbind(Player = c("Euronics", "Mediaworld", "Unieuro"), flex_table_stores)
flex_table_stores


set_flextable_defaults(
  font.family = "Arial", font.size = 10, 
  border.color = "gray", big.mark = "")

ft <- flextable(head(flex_table_stores)) |> 
  bold(part = "header") 
ft

ft |>
  bg(j = "media_rating", 
     bg = scales::col_quantile(palette = c("wheat", "red"), domain =NULL)) |> 
  add_footer_lines("God help us. R is not that nice. Almost as bad as SQL")

# TABELLA DRIVER SENTIMENT

DriverAnalysis <- DriverAnalysis[is.na(DriverAnalysis$SemiSupervised) == FALSE,]

df_drive_recensioni<- DriverAnalysis %>%
  group_by(SemiSupervised) %>%
  summarise(Numero_recensioni = n())

df_drive_sentiment <- DriverAnalysis %>%
  group_by(SemiSupervised) %>%
  summarise(Media_sentiment = mean(sentimentAnalysis))

df_drive_rating <- DriverAnalysis %>%
  group_by(SemiSupervised) %>%
  summarise(Media_rating = mean(score_rating, na.rm = TRUE))

rm(df_drive_rating)

df_drive_tab <- full_join(DriverAnalysis, df_drive_sentiment, by = "SemiSupervised") %>%
  full_join(df_drive_rating, by = "SemiSupervised") %>%
  full_join(df_drive_recensioni, by = "SemiSupervised")

# TABELLA DRIVER RATING

# GRAFICI

# check
table(Ita_StoresReview$score_rating, Ita_StoresReview$NB_sentiment)



# DISTRIBUZIONE SENTIMENT PER BRAND
Brand_sentiment <- as.data.frame(table(Ita_StoresReview$Player, Ita_StoresReview$NB_sentiment))

Brand_sentiment <- rename(
  Brand_sentiment, 
  "Brand" = "Var1", 
  "Sentiment" = "Var2"
)

ggplot(Brand_sentiment,aes(x = Brand, y = Freq, fill = Sentiment))+
  geom_bar(position="stack",stat="identity") +   
  
  scale_fill_manual(values = c("#993333", "grey", "darkseagreen")) +
  labs(title = "Come varia il sentiment nei diversi brand?") +
  coord_flip() +
  ylab(label="Valori assoluti") + 
  xlab("") +
  #la legenda viene generata in modo automatico
  guides(fill=guide_legend(title="Sentiment")) + 
  theme(plot.title = element_text(color = "black", size = 12, face = "bold"),
        plot.subtitle = element_text(face = "plain"),
        axis.title=element_text(size=10,face="plain"),
        axis.text= element_text(size =10, face = "italic"),
        axis.text.x = element_text(color="#993333", angle=45))


# DISTRIBUZIONE RATING PER BRAND
Brand_rating <- as.data.frame(table(Ita_StoresReview$Player, Ita_StoresReview$score_rating))

Brand_rating <- rename(
  Brand_rating, 
  "Brand" = "Var1", 
  "Rating" = "Var2"
)

ggplot(Brand_rating,aes(x = Brand, y = Freq, fill = Rating))+
  geom_bar(position="dodge",stat="identity") +   
  
  scale_fill_manual(values = c("#993333", "#FF5733", "#FFC300", "#DAF7A6", "#7FFF00")) +
  labs(title = "Come varia il rating nei diversi brand?") +
  #rappresentiamo le barre in orizzontale (inverte gli assi)
  #coord_flip() +
  ylab(label="Valori assoluti") + 
  xlab("") +
  #la legenda viene generata in modo automatico
  guides(fill=guide_legend(title="Rating")) + 
  theme(plot.title = element_text(color = "black", size = 12, face = "bold"),
        plot.subtitle = element_text(face = "plain"),
        axis.title=element_text(size=10,face="plain"),
        axis.text= element_text(size =10, face = "italic"),
        axis.text.x = element_text(color="#993333", angle=45))



# RELAZIONE TRA SENTIMENT E RATING
cross_tab <- table(Ita_StoresReview$NB_sentiment, Ita_StoresReview$score_rating)

# Creazione di un grafico a matrice di confusione
heatmap(cross_tab, col = colorRampPalette(c("blue", "white", "red"))(256), main = "Matrice di confusione tra Sentiment e Rating")
legend("topright", legend = c("Basso", "Medio", "Alto"), fill = c("blue", "white", "red"), title = "Rating")

# GRAFICI 

apply(DriverAnalysis, 2, function(x) sum(is.na(x)))


Mediaworld_driver <- DriverAnalysis[DriverAnalysis$Player == "Mediaworld",c(14,15)]
Unieuro_driver <- DriverAnalysis[DriverAnalysis$Player == "Unieuro",c(14,15)]
Euronics_driver <- DriverAnalysis[DriverAnalysis$Player == "Euronics",c(14,15)]
Mediaworld_driver <- as.data.frame(table(Mediaworld_driver$sentiment_labels, Mediaworld_driver$SemiSupervised))
Unieuro_driver <- as.data.frame(table(Unieuro_driver$sentiment_labels, Unieuro_driver$SemiSupervised))
Euronics_driver <- as.data.frame(table(Euronics_driver$sentiment_labels, Euronics_driver$SemiSupervised))

Mediaworld_driver <- rename(
  Mediaworld_driver, 
  "Sentiment" = "Var1", 
  "Driver" = "Var2"
)

Unieuro_driver <- rename(
  Unieuro_driver, 
  "Sentiment" = "Var1", 
  "Driver" = "Var2"
)

Euronics_driver <- rename(
  Euronics_driver, 
  "Sentiment" = "Var1", 
  "Driver" = "Var2"
)

Mediaworld_grafico <- ggplot(Mediaworld_driver,aes(x = Driver, y = Freq, fill = Sentiment))+
  geom_bar(position="stack",stat="identity") +   
  
  scale_fill_manual(values = c("#993333", "darkseagreen")) +
  labs(title = "Mediaworld",
       subtitle = "Come varia il sentiment nei diversi driver?") +
  ylab(label="Valori assoluti") + 
  xlab("") +
  #la legenda viene generata in modo automatico
  guides(fill=guide_legend(title="Sentiment")) + 
  theme(plot.title = element_text(color = "black", size = 12, face = "bold"),
        plot.subtitle = element_text(face = "plain"),
        axis.title=element_text(size=10,face="plain"),
        axis.text= element_text(size =10, face = "italic"),
        axis.text.x = element_text(color="#993333", angle=45))

Unieuro_grafico <- ggplot(Unieuro_driver,aes(x = Driver, y = Freq, fill = Sentiment))+
  geom_bar(position="stack",stat="identity") +   
  
  scale_fill_manual(values = c("#993333", "darkseagreen")) +
  labs(title = "Unieuro",
       subtitle = "Come varia il sentiment nei diversi driver?") +
  ylab(label="Valori assoluti") + 
  xlab("") +
  #la legenda viene generata in modo automatico
  guides(fill=guide_legend(title="Sentiment")) + 
  theme(plot.title = element_text(color = "black", size = 12, face = "bold"),
        plot.subtitle = element_text(face = "plain"),
        axis.title=element_text(size=10,face="plain"),
        axis.text= element_text(size =10, face = "italic"),
        axis.text.x = element_text(color="#993333", angle=45))

Euronics_grafico <- ggplot(Euronics_driver,aes(x = Driver, y = Freq, fill = Sentiment))+
  geom_bar(position="stack",stat="identity") +   
  
  scale_fill_manual(values = c("#993333", "darkseagreen")) +
  labs(title = "Euronics",
       subtitle = "Come varia il sentiment nei diversi driver?") +
  ylab(label="Valori assoluti") + 
  xlab("") +
  #la legenda viene generata in modo automatico
  guides(fill=guide_legend(title="Sentiment")) + 
  theme(plot.title = element_text(color = "black", size = 12, face = "bold"),
        plot.subtitle = element_text(face = "plain"),
        axis.title=element_text(size=10,face="plain"),
        axis.text= element_text(size =10, face = "italic"),
        axis.text.x = element_text(color="#993333", angle=45))


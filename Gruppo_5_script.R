# LIBRERIE ----

# Dataset
library(readxl)
library(writexl)
# Directory
library(rstudioapi)
# Pre-processing
library(quanteda)
library(quanteda.textstats)
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

# 1: DATASET E PULIZIA ----
# Directory della cartella condivisa
setwd(dirname(getActiveDocumentContext()$path))
# Dataset
StoresReview <- read_excel("GRUPPO 3-4-5. Industry elettronica.xlsx")
# Aggiunta Primary key a sinistra del dataframe
StoresReview <- cbind(ID = seq(1:nrow(StoresReview)), StoresReview)

table(StoresReview$lang_value) # presenza di altre lingue

# Eliminiamo le recensioni vuote e manteniamo solo quelle in lingua italiana.
Ita_StoresReview <- StoresReview[(StoresReview$lang_value == "it" |
                                    is.na(StoresReview$lang_value) == TRUE) & 
                                   is.na(StoresReview$text) == FALSE,]
# Putroppo l'algoritmo di deeplearning ha assegnato valori in lang_value 
# diversi da it e NA alle recensioni in italiano,
# Quindi questo filtro li elimina.

table(Ita_StoresReview$social)
# Si nota che nel dataset, le recensioni provengono solo da twitter e places.

# Creazione corpus
Corpus_Totale <- corpus(Ita_StoresReview)

# 2: ANALISI DEL SENTIMENT CON GLI ALGORITMI ----

# Campionamento con numerosità 200
# La qualità delle recensioni di places è superiore rispetto a quelle di twitter,
# Quindi abbiamo deciso di suddivere le due tipologie, prendendo 80% da places
# e il 20 % da twitter
set.seed(001)
Training_places <- sample(Corpus_Totale[attr(Corpus_Totale, "docvars")$social == "places"],
                          size = 160,
                          replace = FALSE)
set.seed(002)
Training_tweet <- sample(Corpus_Totale[attr(Corpus_Totale, "docvars")$social == "twitter"],
                         size = 40,
                         replace = FALSE)

# TRAINING DATA

# Corpus per l'analisi manuale
Training_data <- c(Training_tweet, Training_places)

# Corpus per il TEST SET
Test_data <- Corpus_Totale[!(Corpus_Totale %in% Training_data)]

# Verifica se sono complementari
setequal(Corpus_Totale, union(Test_data, Training_data))
# Risposta affermativa, ma si nota una differenza di 21 testi

# Dataset del Campione per poterlo esportare

Campione <- data.frame(
  attr(Training_data, "docvars")$ID,
  Persona <- rep(c("William","Davide","Maddalena","Giacomo"),each = 50),
  Training_data,
  Sentiment <- NA)
names(Campione) <- c("ID","Persona","text","sentiment")

# dataframe del test set
Test_data <- data.frame(
  attr(Test_data,"docvars")$ID,
  Test_data
)
names(Test_data) <- c("ID","text")

#Esportare il Campione
#write_xlsx(Campione, "Training Data Grezzo.xlsx") # NON RUNNARE !!!!!!!!!!!!!!!!!!!!!!!
Campione <- read_excel("Training Data Grezzo.xlsx")

Campione$sentiment <- ifelse(Campione$sentiment == -1, "Negativo", 
                             ifelse(Campione$sentiment == 0, "Neutro", 
                                    "Positivo"))

# Verifica celle vuote.
apply(Campione, 2, function(x) sum(is.na(x)))

# Conversione in corpus con la variabile del sentiment
Training_data <- corpus(Campione)



Dfm_Training <- dfm(tokens(Training_data,
                           remove_punct = TRUE,
                           remove_symbols = TRUE,
                           remove_url = TRUE,
                           remove_numbers = TRUE) %>%
                      tokens_tolower() %>% 
                      tokens_remove(c(stopwords("italian"))) %>%
                      tokens_wordstem(language = "italian"))

Dfm_Test <- dfm(tokens(Test_data,
                       remove_punct = TRUE,
                       remove_symbols = TRUE,
                       remove_url = TRUE,
                       remove_numbers = TRUE) %>%
                  tokens_tolower() %>% 
                  tokens_remove(c(stopwords("italian"))) %>%
                  tokens_wordstem(language = "italian"))

length(Dfm_Training@Dimnames$features) #1251
length(Dfm_Test@Dimnames$features) #8576

# Abbasso il numero di features del test
Dfm_Test <- dfm_match(Dfm_Test, 
                      features = featnames(Dfm_Training))
# Dopo il match lunghezzze pari a 1251

# Verifica
setequal(featnames(Dfm_Training), 
         featnames(Dfm_Test)) 
# Creazione matrici per gli algoritmi
Matrice_Training <- as.matrix(Dfm_Training)
Matrice_Test <- as.matrix(Dfm_Test)

# Conversione del vettore sentiment in factor
Dfm_Training@docvars$sentiment <- as.factor(Dfm_Training@docvars$sentiment)

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

# ALGORITMO RANDOM FOREST
set.seed(150)
RF <- randomForest(y= Dfm_Training@docvars$sentiment,
                   x= Matrice_Training,
                   importance=TRUE,
                   do.trace=FALSE,
                   ntree=500)
RF$predicted

RF
table(Campione$sentiment)

plot(RF, type = "l", col = c("black", "steelblue4","violetred4", "springgreen4"),
     main = "Random Forest Model Errors: sentiment variable")
legend("topright", horiz = F, cex = 0.7,
       fill = c("springgreen4", "black", "steelblue4", "violetred4"),
       c("Positive error", "Average error", "Negative error", "Neutral error"))

Errori <- as.data.frame(RF$err.rate)

which.min(Errori$OOB) # 38

set.seed(150)
RF <- randomForest(y= Dfm_Training@docvars$sentiment,
                   x= Matrice_Training,
                   importance=TRUE,
                   do.trace=FALSE,
                   ntree=38)

Test_predictedRF <- predict(RF, Matrice_Test ,type="class")

# SUPPORT VECTOR
set.seed(175)
SupportVectorMachine <- svm(
  y= Dfm_Training@docvars$sentiment,
  x=Matrice_Training, kernel='linear', cost = 1)

Test_predictedSV <- predict(SupportVectorMachine, Matrice_Test)
length(SupportVectorMachine$index)

Test_data$Bayes <- Test_predictedNB
Test_data$Forest <- Test_predictedRF
Test_data$Support <- Test_predictedSV

str(Test_data)
results <- as.data.frame(rbind(prop.table(table(Test_predictedNB)),
                               prop.table(table(Test_predictedRF)),
                               prop.table(table(Test_predictedSV))))

results$algorithm <- c("Naive Bayes", "Random Forest", "Support Vector Machine")

df.long<-melt(results,id.vars=c("algorithm"))
str(df.long)

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

Matrice_Training2 <- Matrice_Training

#Assicuiamo la replicabilità 
set.seed(200)
#Definiamo un oggetto k che indichi il numero di folders
k <- 5
#Dividiamo la matrice in k folders
folds <- cvFolds(NROW(Matrice_Training2), K = k)

system.time(for(i in 1:k){
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
})

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

#guardiamo la struttura 
str(NB_Prediction)
NB_Prediction [is.na(NB_Prediction )] <- 0

AverageAccuracy_NB <- mean(NB_Prediction[, 1] )
AverageF1_NB<- mean(colMeans(NB_Prediction[-1] ))
AverageAccuracy_NB
AverageF1_NB

# RANDOM FOREST
system.time(for(i in 1:k){
  Matrice_Training <-
    Matrice_Training2 [folds$subsets[folds$which != i], ]
  ValidationSet <-
    Matrice_Training2 [folds$subsets[folds$which == i], ]
  set.seed(250)
  RandomForest <- randomForest(
    y= Dfm_Training[folds$subsets[folds$which != i], ]
    @docvars$sentiment ,
    x=Matrice_Training, do.trace=FALSE, ntree=53)
  Predictions_RF <- predict(RandomForest, 
                            newdata= ValidationSet, 
                            type="class")
  class_table <- table("Predictions"= Predictions_RF,
                       "Actual"=Dfm_Training[folds$subsets[folds$which == i], ]@docvars$sentiment)
  print(class_table)
  df<-confusionMatrix( class_table, mode = "everything")
  df_measures_RF<-paste0("conf.mat.rf",i)
  assign(df_measures_RF,df)
})

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

str(RF_Predictions)

RF_Predictions [is.na(RF_Predictions )] <- 0

#Calcoliamo i valori medi
AverageAccuracy_RF <- mean(RF_Predictions[, 1] )
AverageF1_RF<- mean(colMeans(RF_Predictions[-1] ))

AverageAccuracy_RF
AverageF1_RF


# SUPPORT VECTOR MACHINE

system.time(for(i in 1:k){
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
})


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

str(SVM_Prediction)
SVM_Prediction [is.na(SVM_Prediction)] <- 0


#Calcoliamo i valori medi
AverageAccuracy_SVM <- mean(SVM_Prediction[, 1] )
AverageF1_SVM<- mean(colMeans(SVM_Prediction[-1] ))

AverageAccuracy_SVM
AverageF1_SVM


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
str(Accuracy_models_Melt)

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
# 3: DRIVER ANALYSIS ----

# Corpus

# Frequenze delle caratteristiche del Corpus
apply(textstat_summary(Corpus_Totale)[,2:11], 2, sum)

# DFM... MODIFICARE LE CONDIZIONI TRIMMING
Dfm_Totale <- dfm(tokens(Corpus_Totale,
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_url = TRUE,
                        remove_numbers = TRUE) %>%
                   tokens_tolower() %>% 
                   tokens_remove(c(stopwords("italian"))) %>%
                   tokens_wordstem(language = "italian")) %>%
              dfm_trim(min_termfreq = 10,
                       max_termfreq = 500,
                       min_docfreq = 2)

Dfm_Places <- dfm(tokens(Corpus_Totale[attr(Corpus_Totale, "docvars")$social == "places"],
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_url = TRUE,
                         remove_numbers = TRUE) %>%
                    tokens_tolower() %>% 
                    tokens_remove(c(stopwords("italian"))) %>%
                    tokens_wordstem(language = "italian")) %>%
  dfm_trim(min_termfreq = 10,
           max_termfreq = 500,
           min_docfreq = 2)
  
#FARE WORDCLOUD per le keywords e scelta di variabili CATEGORIALI
# Raggruppare i valori delle varia colonne, in base al brand e alla presenza di keywords relative alla variabile categoriale scelta
# QUINDI CRARE UN DATAFRAME CON I VALORI AGGREGATI

# Toglie i tag e gli hashtag
Parole_Brutte <- colnames(Dfm_Totale)[grepl("^\\s*[#@]", trimws(colnames(Dfm_Totale)))]
Dfm_Totale <- Dfm_Totale[,!(colnames(Dfm_Totale) %in% Parole_Brutte)]


# RILEVAZIONE DELLE KEYWORDS

# Top parole del DFM
topfeatures(Dfm_Places,50)

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

Contenitore <- as.data.frame(colnames(Dfm_Totale))
write_xlsx(Contenitore, "Lista parole.xlsx")
#write_xlsx(Campione, "Training Data Grezzo.xlsx") 

Driver <- dictionary(list(Prezzo = c("offert*","scont*","prezz*","vend*","cost*","sottocost*", "economic*"),
                          Servizio = c("personal*","serviz*","gentil*","professional*","competent*","aiut*","cortes*","assistent*","disponibil*","cordial*",
                                       "scortes*","male*","lent*","disorg*","disorie*"),
                          Ordini = c("ordin*","consegn*","ritir*","garanz*","online*","spedi*","reso","account"),
                          Location = c("negoz*","post*","parchegg*","affollat*","piccol*","disord*")))

Driver_Review <- dfm_lookup(Dfm_Totale,Driver)
Driver_Review

Driver_Conv_Rev <- convert(Driver_Review, to = "data.frame")

apply(Driver_Conv_Rev[,2:5],2,sum)
prop.table(apply(Driver_Conv_Rev[,2:5],2,sum)) # da sistemare
# Creare il grafico apposito nella scelta delle keywords per vedere in quali recensioni appaiono

# Estrarre un campione di 150 e calcolare la % di uguaglianza


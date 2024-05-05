# 1: Pulizia e Preparazione dei dati ----
install.packages("naivebayes")
library(readxl)
library(writexl)
library(rstudioapi)
library(quanteda)
library(quanteda.textstats)
library(naivebayes)
library(ggplot2)



# Directory della cartella condivisa
setwd(dirname(getActiveDocumentContext()$path))
# Dataset
StoresReview <- read_excel("GRUPPO 3-4-5. Industry elettronica.xlsx")
# Aggiunta Primary key
StoresReview$ID <- seq(1:nrow(StoresReview))
# Dataset con sole recensioni in italiano.
Ita_StoresReview <- StoresReview[(StoresReview$lang_value == "it" |
                                    is.na(StoresReview$lang_value) == TRUE) & 
                                   is.na(StoresReview$text) == FALSE,]


# PRE- PROCESSING DFM ----

# Corpus con i testi NON vuoti
Corpus_Totale <- corpus(Ita_StoresReview)
attr(Corpus_Totale, "docvars")$ID

# Frequenze delle caratteristiche del Corpus
apply(textstat_summary(Corpus_Totale)[,2:11], 2, sum)

# NON PULISCE TUTTO. !!, emoji
Dfm_Totale <- dfm(tokens(Corpus_Totale,
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_url = TRUE,
                        remove_numbers = TRUE) %>%
                   tokens_tolower() %>% 
                   tokens_remove(c(stopwords("italian"))) %>%
                   tokens_wordstem(language = "italian")) %>%
              dfm_trim(min_termfreq = 10,
                       min_docfreq = 2)

Dfm_Totale@docvars$ID
# Lunghezza del DFM
summary(Dfm_Totale)
# Top parole del DFM
topfeatures(Dfm_Totale,100)


# ANALISI ----
# Suddivisione dataset per social
Tweet_ita <- Ita_StoresReview[Ita_StoresReview$social == "twitter",]
Places_ita <- Ita_StoresReview[Ita_StoresReview$social == "places",]
# Controllo testi vuoti
apply(Tweet_ita, 2, function(x) sum(is.na(x))) # 0 testi NA
apply(Places_ita, 2, function(x) sum(is.na(x))) # 458 testi NA!!

# Corpus per i Tweet
Tweet_Corpus <- corpus(Tweet_ita)

# Foreign key impostate
attr(Tweet_Corpus, "docvars")$ID
# Corpus per i Places
Places_Corpus <- corpus(Places_ita)
# Foreign key impostate
attr(Places_Corpus, "docvars")$ID

# Campionamento con numerosità 200
set.seed(001)
Training_places <- sample(Places_Corpus, size = 160, replace = FALSE)
set.seed(002)
Training_tweet <- sample(Tweet_Corpus, size = 40, replace = FALSE)

attr(Training_tweet, "docvars")$ID
# TRAINING DATA

docnames(Training_tweet) <- paste0("new_", docnames(Training_tweet))
Training_data <- c(Training_tweet, Training_places)

# Corpus per il TEST SET
Test_data <- Corpus_Totale[!(Corpus_Totale %in% Training_data)]
attr(Test_data, "docvars")$ID
# Verifica complementare
setequal(Corpus_Totale, union(Test_data, Training_data))

# Dataset del Campione

Campione <- data.frame(
  attr(Training_data, "docvars")$ID,
  Persona <- rep(c("William","Davide","Maddalena","Giacomo"),each = 50),
  Training_data,
  Sentiment <- NA)
names(Campione) <- c("ID","Persona","text","sentiment")

Test_data <- data.frame(
  attr(Test_data,"docvars")$ID,
  Test_data
)
names(Test_data) <- c("ID","text")

#Esportare il Campione
write_xlsx(Campione, "Training Data Grezzo.xlsx") # NON RUNNARE !!!!!!!!!!!!!!!!!!!!!!!
Campione <- read_excel("Training Data Grezzo.xlsx")

Campione$sentiment <- ifelse(Campione$sentiment == -1, "Negativo", 
                                  ifelse(Campione$sentiment == 0, "Neutro", 
                                        "Positivo"))

# Verifica celle vuote.
apply(Campione, 2, function(x) sum(is.na(x)))
# Nome text per il text_field

str(Training_data)
Training_data <- corpus(Campione)
Test_Corpus <- corpus(Test_data)


Dfm_Training <- dfm(tokens(Training_data,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_url = TRUE,
                         remove_numbers = TRUE) %>%
                    tokens_tolower() %>% 
                    tokens_remove(c(stopwords("italian"))) %>%
                    tokens_wordstem(language = "italian")) %>%
                dfm_trim(min_termfreq = 10,
                         min_docfreq = 2)

Dfm_Test <- dfm(tokens(Test_Corpus,
                           remove_punct = TRUE,
                           remove_symbols = TRUE,
                           remove_url = TRUE,
                           remove_numbers = TRUE) %>%
                      tokens_tolower() %>% 
                      tokens_remove(c(stopwords("italian"))) %>%
                      tokens_wordstem(language = "italian")) %>%
                dfm_trim(min_termfreq = 10,
                         min_docfreq = 2)

length(Dfm_Training@Dimnames$features) #61
length(Dfm_Test@Dimnames$features) #867

Dfm_Test <- dfm_match(Dfm_Test, 
                      features = featnames(Dfm_Training))

setequal(featnames(Dfm_Training), 
         featnames(Dfm_Test)) 

# Dopo il match lunghezzze pari a 61

Matrice_Training <- as.matrix(Dfm_Training)
Matrice_Test <- as.matrix(Dfm_Test)


str(Dfm_Training@docvars$sentiment)
Dfm_Training@docvars$sentiment <- as.factor(Dfm_Training@docvars$sentiment)

set.seed(123) 

system.time(NaiveBayesModel <- multinomial_naive_bayes
            (x=Matrice_Training,
              y=Dfm_Training@docvars$sentiment,
              laplace = 1))
summary(NaiveBayesModel)

Test_predictedNB <- predict(NaiveBayesModel,
                            Matrice_Test)
# Check list ----
str(Test_predictedNB)
# Scrivere qui tutti gli step fatti e da fare
Tabella_descrittiva <- textstat_frequency(Testo_finito, n =500)



# SUGGERIMENTI ----

# allenamento dell'algoritmo.
# estrazione campionaria randomica
# 200 testi a testa
# codebook, documento word per noi
# driver serve il dizionario (normale o newsmap).

# 1: Pulizia e Preparazione dei dati ----

library(readxl)
library(writexl)
library(rstudioapi)
library(quanteda)
library(quanteda.textstats)
library(ggplot2)



# Directory della cartella condivisa
setwd(dirname(getActiveDocumentContext()$path))
# Dataset
StoresReview <- read_excel("GRUPPO 3-4-5. Industry elettronica.xlsx")
# Aggiunta Primary key
StoresReview$ID <- seq(1:nrow(StoresReview))
# Dataset con sole recensioni in italiano.
Ita_StoresReview <- StoresReview[StoresReview$lang_value == "it" | is.na(StoresReview$lang_value) == TRUE,]


# PRE- PROCESSING DFM ----

# Corpus con i testi NON vuoti
Corpus_Totale <- corpus(na.omit(Ita_StoresReview$text))
names(Corpus_Totale) <- Ita_StoresReview$ID[is.na(Ita_StoresReview$text) == FALSE]

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
                   tokens_wordstem(language = "italian"))
# Applicazione del TRIMMING
Dfm_Totale <- dfm_trim(Dfm_Totale,
                        min_termfreq = 10,
                        min_docfreq = 2)
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
# Elimino le recensioni vuote
Places_ita <- Places_ita[is.na(Places_ita$text) == FALSE,]

# Corpus per i Tweet
Tweet_Corpus <- corpus(Tweet_ita)
# Foreign key impostate
names(Tweet_Corpus) <- Tweet_ita$ID
# Corpus per i Places
Places_Corpus <- corpus(Places_ita)
# Foreign key impostate
names(Places_Corpus) <- Places_ita$ID 

# Campionamento con numerosità 200
set.seed(001)
Training_places <- sample(Places_Corpus, size = 160, replace = FALSE)
set.seed(002)
Training_tweet <- sample(Tweet_Corpus, size = 40, replace = FALSE)

# TRAINING DATA
Training_data <- c(Training_tweet, Training_places)

# Corpus per il TEST SET
Review_test <- Corpus_Totale[!(Corpus_Totale %in% Training_data)]

# Verifica complementare
setequal(Corpus_Totale, union(Review_test, Training_data))

# Dataset del Campione
Campione <- data.frame(
  ID <- names(Training_data),
  Persona <- rep(c("William","Davide","Maddalena","Giacomo"),each = 50),
  Testo <- Training_data,
  Sentiment <- NA
)

Review_test_1 <- data.frame(
  ID <- names(Review_test),
  Review_test
)

#Esportare il Campione
# write_xlsx(Campione, "Lavoro.xlsx") # NON RUNNARE !!!!!!!!!!!!!!!!!!!!!!!
Campione <- read_excel("Lavoro.xlsx")

Campione$Sentiment <- ifelse(Campione$Sentiment == -1, "Negativo", 
                                  ifelse(Campione$Sentiment == 0, "Neutro", 
                                        "Positivo"))

# Verifica celle vuote.
apply(Campione, 2, function(x) sum(is.na(x)))
# Nome text per il text_field
colnames(Campione) <- c("ID", "Persona", "text","Sentiment")
colnames(Review_test_1) <- c("ID", "text")

Campione_Corpus <- corpus(Campione)
Review_test_1 <- corpus(Review_test_1)

Dfm_Training <- dfm(tokens(Campione_Corpus,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_url = TRUE,
                         remove_numbers = TRUE) %>%
                    tokens_tolower() %>% 
                    tokens_remove(c(stopwords("italian"))) %>%
                    tokens_wordstem(language = "italian"))

Dfm_Test <- dfm(tokens(Review_test_1,
                       remove_punct = TRUE,
                       remove_symbols = TRUE,
                       remove_url = TRUE,
                       remove_numbers = TRUE) %>%
                  tokens_tolower() %>% 
                  tokens_remove(c(stopwords("italian"))) %>%
                  tokens_wordstem(language = "italian"))

Dfm_Training <- dfm_trim(Dfm_Training,
                         min_termfreq = 10,
                         min_docfreq = 2)

Dfm_Test <- dfm_trim(Dfm_Test,
                     min_termfreq = 10,
                     min_docfreq = 2)

length(Dfm_Training@Dimnames$features) #61
length(Dfm_Test@Dimnames$features) #867

setequal(featnames(Dfm_Training), 
         featnames(Dfm_Test)) 


Dfm_Test <- dfm_match(Dfm_Test, 
                      features = featnames(Dfm_Training))
# Dopo il match lunghezzze pari a 61

Matrice_Training <- as.matrix(Dfm_Training)
Matrice_Test <- as.matrix(Dfm_Test)

str(Dfm_Training@docvars$Sentiment) #impostare le minuscole
Dfm_Training@docvars$Sentiment <- as.factor(Dfm_Training@docvars$Sentiment)
# Check list ----

# Scrivere qui tutti gli step fatti e da fare
Tabella_descrittiva <- textstat_frequency(Testo_finito, n =500)


# SUGGERIMENTI ----

# allenamento dell'algoritmo.
# estrazione campionaria randomica
# 200 testi a testa
# codebook, documento word per noi
# driver serve il dizionario (normale o newsmap).

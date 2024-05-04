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
                        #max_termfreq = 500,
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
Places_ita <- Places_ita[is.na(Places_ita$text) == FALSE,] # 0 testi NA.


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
setequal(Corpus_Totale, union(Review_test, Campione))

Campione <- data.frame(
  ID <- names(Training_data),
  Persona <- rep(c("William","Davide","Maddalena","Giacomo"),each = 50),
  Testo <- Training_data,
  Sentiment <- NA
)

write_xlsx(Campione, "Lavoro.xlsx")

# Semplice sequenza di codice per esportare / importare ed editare su excel  --------

#Export
df = data.frame(names(Lavoro$Davide), Lavoro$Davide, NA) # put your fucking name
colnames(df) <- c('TextNumber', 'text', 'sentiment')

write.xlsx(df, "davidino.xlsx") # put your fucking name

# Import
df <- as.data.frame(read_excel("davidino.xlsx")) # put your fucking name
# E così importate automaticamente il vostro lavoro fatto come un dataframe. 


# Check list ----

# Scrivere qui tutti gli step fatti e da fare
Tabella_descrittiva <- textstat_frequency(Testo_finito, n =500)

# SUGGERIMENTI ----

# allenamento dell'algoritmo.
# estrazione campionaria randomica
# 200 testi a testa
# codebook, documento word per noi
# driver serve il dizionario (normale o newsmap).

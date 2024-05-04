# librerie e dataset ----
library(readxl)
library(rstudioapi)
library(quanteda)
library(quanteda.textstats)
library(ggplot2)

setwd(dirname(getActiveDocumentContext()$path))
# Questo codice magico permette di settare automaticamente la working directory nella cartella in cui si trova lo script
# è utile perchè, lavorando con git, cloniamo continuamente e non si può stare a cambiare ogni volta il path per ogni pc diverso

StoresReview <- read_excel("GRUPPO 3-4-5. Industry elettronica.xlsx")

# Dataset con sole recensioni in italiano.
Ita_StoresReview <- StoresReview[StoresReview$lang_value == "it" | is.na(StoresReview$lang_value) == TRUE,]

# PRE- PROCESSING DFM ----

# Creazione del Corpus prendendo solo i testi NON vuoti
Testo_Corpus <- corpus(na.omit(Ita_StoresReview$text))

# Check
textstat_summary(Testo_Corpus)


# DFM - VERSIONE 1: risultato 35 MILIONI
Testo_dfm <- dfm(tokens(Testo_Corpus,
                        remove_numbers = TRUE,
                        remove_punct = TRUE,
                        remove_symbols = TRUE) %>%
                   tokens_tolower() %>% 
                   tokens_remove(c(stopwords("italian"))) %>%
                   tokens_wordstem(language = "italian"))

# DFM - VERSIONE 2: risultato 32 MILIONI
# NON PULISCE TUTTO. !!, emoji
Testo_dfm <- dfm(tokens(Testo_Corpus,
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_url = TRUE,
                        remove_numbers = TRUE) %>%
                   tokens_tolower() %>% 
                   tokens_remove(c(stopwords("italian"))) %>%
                   tokens_wordstem(language = "italian"))
# Check
summary(Testo_dfm)

# Applicazione del TRIMMING: condizioni TEMPORANEE.
Testo_finito <- dfm_trim(Testo_dfm,
                        min_termfreq = 10,
                        #max_termfreq = 500,
                        min_docfreq = 2)

topfeatures(Testo_finito,100)


# ANALISI ----

# Per grafici sulle keywords
Tabella_descrittiva <- textstat_frequency(Testo_finito, n =500)

#Campionamento per il TRAINING STAGE
set.seed(000)
Review_training <- sample(Testo_Corpus, size = 200, replace = FALSE)

# Corpus per il TEST SET
Review_test <- Testo_Corpus[!(Testo_Corpus %in% Review_training)]

# Verifica Complementari
setequal(Testo_Corpus, union(Review_test, Review_training))

# Runnare da 71 a 82 per Riempire la lista
Lavoro <- list(
  William = rep("", 50),
  Davide = rep("", 50),
  Maddalena = rep("", 50),
  Giacomo = rep("", 50)
)
k <- 0

for (i in 1:4){
  Lavoro[[i]] <- Review_training[(k+1) : (50 * i)]
  k <- 50 * i
}


# Semplice sequenza di codice per esportare / importare ed editare su excel  --------
library(openxlsx)
#Export
df = data.frame(names(Lavoro$Davide), Lavoro$Davide, NA) # put your fucking name
colnames(df) <- c('TextNumber', 'text', 'sentiment')

write.xlsx(df, "davidino.xlsx") # put your fucking name

# Import
df <- as.data.frame(read_excel("davidino.xlsx")) # put your fucking name
# E così importate automaticamente il vostro lavoro fatto come un dataframe. 


# Check list ----

# Scrivere qui tutti gli step fatti e da fare


# SUGGERIMENTI ----

# allenamento dell'algoritmo.
# estrazione campionaria randomica
# 200 testi a testa
# codebook, documento word per noi
# driver serve il dizionario (normale o newsmap).

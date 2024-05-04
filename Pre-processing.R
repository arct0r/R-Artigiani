# librerie e dataset ----
library(readxl)
library(writexl)
library(rstudioapi)
library(quanteda)
library(quanteda.textstats)
library(ggplot2)

setwd(dirname(getActiveDocumentContext()$path))
# Questo codice magico permette di settare automaticamente la working directory nella cartella in cui si trova lo script
# è utile perchè, lavorando con git, cloniamo continuamente e non si può stare a cambiare ogni volta il path per ogni pc diverso

StoresReview <- read_excel("GRUPPO 3-4-5. Industry elettronica.xlsx")
StoresReview$ID <- seq(1:nrow(StoresReview))

# Dataset con sole recensioni in italiano.
Ita_StoresReview <- StoresReview[StoresReview$lang_value == "it" | is.na(StoresReview$lang_value) == TRUE,]

# PRE- PROCESSING DFM ----

# Creazione del Corpus prendendo solo i testi NON vuoti
Corpus_Totale <- corpus(na.omit(Ita_StoresReview$text))


# Check
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
# Check
summary(Dfm_Totale)

# Applicazione del TRIMMING: condizioni TEMPORANEE.
DFM_Totale_Finito <- dfm_trim(Dfm_Totale,
                        min_termfreq = 10,
                        #max_termfreq = 500,
                        min_docfreq = 2)

topfeatures(DFM_Totale_Finito,100)


# ANALISI ----

# Per grafici sulle keywords
Tabella_descrittiva <- textstat_frequency(Testo_finito, n =500)

Tweet_ita <- Ita_StoresReview[Ita_StoresReview$social == "twitter",]
Places_ita <- Ita_StoresReview[Ita_StoresReview$social == "places",]
Places_ita <- Places_ita[is.na(Places_ita$text) == FALSE,]

apply(Tweet_ita, 2, function(x) sum(is.na(x)))
apply(Places_ita, 2, function(x) sum(is.na(x)))



 
Tweet_Stores <- Ita_StoresReview[Ita_StoresReview$social == "twitter",]
Places_Stores <- Ita_StoresReview[Ita_StoresReview$social == "places",]

#Campionamento per il TRAINING STAGE
Tweet_Corpus <- corpus(Tweet_ita)
names(Tweet_Corpus) <- Tweet_ita$ID

Places_Corpus <- corpus(Places_ita)
names(Places_Corpus) <- Places_ita$ID 

set.seed(001)
Training_places <- sample(Places_ita$text, size = 160, replace = FALSE)



set.seed(002)
Training_tweet <- sample(Tweet_ita, size = 40, replace = FALSE)
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

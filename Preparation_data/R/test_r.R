library(readxl)
df <- read_excel('test_r2.xlsx',sheet=1)

#df$Date <- strptime(df$DATE, "%d/%m/%Y %H:%M")

#df$Date <- strptime(df$Date, "%d/%m/%Y %H:%M")
df$WEEKDAYS <- weekdays(df$Date)
write.csv2(df, file="myfile.csv")
#df$MONTHS <- months(df$DATE)
#df$QUARTERS <- quarters(df$DATE)
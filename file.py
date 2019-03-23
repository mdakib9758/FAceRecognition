import pymysql

my=pymysql.connect(host="localhost",user="root",passwd="akkii",db="flipkart")
cur=my.cursor()
cur.execute("show tables")
for x in cur:
    print(x)

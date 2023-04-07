import time
import pymysql
import random

query = ["SELECT * FROM cosmetic;", 
         "SELECT goods_nm, brand_nm, price FROM cosmetic WHERE category='에센스';", 
         "SELECT COUNT(*) AS total_count FROM cosmetic WHERE brand_nm='라운드랩';",
         "SELECT brand_nm, COUNT(*) AS total_count FROM cosmetic GROUP BY brand_nm ORDER BY total_count DESC LIMIT 10;",
         "SELECT goods_nm, brand_nm, price FROM cosmetic WHERE goods_nm LIKE '%스킨%';",
         "SELECT * FROM cosmetic WHERE price <= 10000 ORDER BY created_at DESC LIMIT 1;"]
print(query[0])
# MySQL 연결 정보 입력
connection = pymysql.connect(
    host="localhost",
    user="root",
    database="dev",
    cursorclass=pymysql.cursors.DictCursor,
)

# 실행할 쿼리

# 쿼리 실행 및 평균 응답시간 측정
num_queries = 2000
total_time = 0
for i in range(num_queries):
    start_time = time.time()
    with connection.cursor() as cursor:
        qu = random.choice(query)
        print(qu)
        cursor.execute(qu)
    end_time = time.time()
    total_time += (end_time - start_time)

# 평균 응답시간 출력
avg_time = total_time / num_queries
print(f"Average response time for {num_queries} queries: {avg_time:.4f} seconds")

# 연결 종료
connection.close()



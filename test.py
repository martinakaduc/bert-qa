from infer import infer

context = "Chelsea cần chiến thắng trước Stoke City ở Stamford Bridge để cân bằng kỷ lục 13 chiến thắng của Arsenal ở cuối mùa 2001/02. Cuối cùng, đoàn quân của HLV Conte đã hoàn thành nhiệm vụ này khi vượt qua đối thủ với tỷ số 4-2."
question = "Có hai đội nào trong trận đấu này?"

result = infer([context, context], [question, question])
print(result)

# processor
文档解析

20240618版本
1. text的base_64赋值 “meaningless”
2. IMAGE的text加一个占位符 ![image](attachment:image)
3. 合并chunk时将👆IMAGE的占位符一起合为new_chunk
4. 增加text_list字段收集用于合并的chunk，它可能包含文本和图片的占位符
5. positions里加入image的位置信息
6. base_64字段List[str]：存储base_64列表

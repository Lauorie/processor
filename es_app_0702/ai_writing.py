from config import Config
from prompts import PROMPTS
from flask_cors import CORS
from loguru import logger
from vllm_llm import LLM
from flask import Flask, Response, request, jsonify, stream_with_context

app = Flask(__name__) 
CORS(app)

AI_WRITING_PROMPT = PROMPTS.AI_WRITING_PROMPT

llm = LLM()

def ai_writing(topic, evidence, requirements=None):
    prompt = AI_WRITING_PROMPT.format(topic=topic, evidence=evidence, requirements=requirements)
    logger.info(f"Prompt: {prompt}")
    result = []
    
    @stream_with_context
    def generate():
        for res in llm.chat_stream(prompt):
            yield res
            result.append(res)
        logger.info(f"Result: {''.join(result)}")
    return Response(generate(), mimetype='text/event-stream')
    

@app.route('/ai_writing/V1/', methods=['POST'])
def ai_writing_api():
    data = request.json
    topic = data.get('topic')
    evidence = data.get('evidence')
    requirements = data.get('requirements', None)
    
    # 验证数据
    if not topic:
        return jsonify({"error": "topic is required."}), 400
    if not evidence:
        return jsonify({"error": "evidence is required."}), 400
    try:
        return ai_writing(topic, evidence, requirements)
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "vLLM model error."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.AI_WRITING_PORT, debug=True)
    
    

# if __name__ == "__main__":
#     topic = "农作物种子供需格局稳定，玉米种子制种基地资源紧俏"
#     evidence = """据全国农技中心数据，杂交玉米种子方面，2022 年制种收获面积大幅增加至 366 万亩，创近 6 年历史最高记录。制种基地分布方面，甘肃仍是第一大制种所在区域，2022 年制种面积达 175 万亩。西北基地杂交玉米种子生产成本在 2021 年的基础上再次上涨700-1000 元，其中甘肃制种基地较新疆涨幅更大。玉米种子供种格局方面， 据全国农技中心数据， 2023 年玉米种子供给量约为 16.5亿公斤，需求量约 11.5 亿公斤，供需比约 140%，处于供大于求状态。预计 2023 年期末库存为 4 亿公斤，库存/总需种量为35%，同比增加5pct。预计2023 年东北和西南地区少数早熟和热带血缘品种供应偏紧。杂交水稻方面，据全国农技中心数据，2022 年全国常规稻繁种单产为 519 公斤/亩，同比增加 22 公斤/亩，新产常规稻种 11.8 亿公斤比 2021 年增加2.3 亿公斤、增幅 24%。 2022 年全国杂交稻平均单产为 144 公斤/亩， 同比减少 22%，产量减幅较大。大豆方面，据全国农技中心数据，2022 年全国大豆繁种面积达 511 万亩，同比增长 60 万亩，新产大豆种子约 8.8 亿公斤，同比增加 1.3 亿公斤，增幅达 17%。东北地区作为大豆繁种主产区，2022 年共制种收获393 万亩，同比增加38 万亩，增幅达 11%，大豆繁种基地总体长势良好，仅极少数地区受霉根腐病影响。"""
#     requirements = "杂交玉米种子、杂交水稻、大豆、农作物种子，供需稳定"
    
#     generator = ai_writing(topic, evidence, requirements)
    
#     for res in generator:
#         print(res)

#     # print(f"final_result: {final_result}")
from vllm_llm import LLM
import copy

# prefix instruction
def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

# suffix instruction
def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

# add passages to prompt
def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 512

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.strip()
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def run_llm(messages):
    llm = LLM()
    ans = llm.streaming_answer(messages)
    answer = ''
    for i in ans:
        answer += i
    return answer

def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response

def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    # clean the response
    response = clean_response(permutation)

    # zero index
    response = [int(x) - 1 for x in response.split()]

    # remove duplicates and convert to list
    response = remove_duplicate(response)

    # cut the part of the list of passages you want to re-rank
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])

    # get a zero indexed initial rank of passages from the cut range
    original_rank = [tt for tt in range(len(cut_range))]

    # get only the response ranks which are in the possible range of 
    # the zero indexed original ranks
    response = [ss for ss in response if ss in original_rank]

    # append the items in the original rank that are not in the permutation
    # response
    response = response + [tt for tt in original_rank if tt not in response]
    
    # this is where the re-ranking happens
    for j, x in enumerate(response):
        # move passages to their re-ranked position
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
    return item

# a function to organise everything together
def permutation_pipeline(item=None, rank_start=0, rank_end=100):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)
    permutation = run_llm(messages)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item

def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


# item = {'query': '姚明的身高是多少？',
#   'hits': [{'content': 'Even in the NBA, where tall players are common, he stood out.'},
#    {'content': '即使在长人如林的NBA，他也显得鹤立鸡群。'},
#    {'content': "His presence changed people's stereotypes about Asian basketball."},
#    {'content': '他的出现改变了人们对亚洲篮球的刻板印象。'},
#    {'content': 'His height was a natural barrier on the basketball court.'},
#    {'content': '他需要弯腰才能穿过大多数的门。'},
#    {'content': '他的身高是篮球场上天然的屏障。'},
#    {'content': 'He could almost touch the top of the basketball hoop.'},
#    {'content': '他几乎可以触摸到篮球架的顶端。'},
#    {'content': 'He had to duck to get through most doors.'}]}


docs = ["Lockheed_martin_F35_user_manual-1 (2).pdf的43 ft (13.1 m) Wing Area 460 ft² (42.7m²) 460 ft² (42.7 m²) 668 ft² (62.1 m²) Empty weight 29,098 lb (13,199kg) 32,300 lb (14,700 kg) 34,800 lb (15,800 kg) Internal fuel 18,498 lb (8,390kg) 13,326 lb (6,045 kg) 19,624 lb (8,900 kg) Max takeoff weight 70,000 lb class (31,800 kg) 60,000 lb class (27,300 kg) 70,000 lb class (31,800 kg)的部分内容为:The CTOL variant is intended for the US Air Force and other air forces. It is the smallest, lightest F-35 version and it is the only variant equipped with an internal cannon, the GAU-22/A. The F-35A is expected to match the F-16 in maneuverability, instantaneous and sustained high-g performance, and outperform it in stealth, payload, range on internal fuel, avionics, operational effectiveness, supportability and survivability. It also has an internal laser designator and infrared sensors, equivalent with the Sniper XR pod carried by the F-16, but built in to remain stealthy.\nThe A variant is primarily intended to replace the USAF's F-16 Fighting Falcon, beginning in 2013, and the A10 Thunderbolt II starting in 2028. F-35A, the Conventional Take-Off and Landing version (CTOL) in flightThe CTOL version can be easily recognized by the absence of the big fan door of the STOVL version and by the presence of the gun on the port side of the fuselage and by the retractable refuel receptacle.\nThe peculiar characteristics of the -A version are:\nIt is the lightest and the fastest version", '问界m9-product-manual.pdf的前言 3 目录的部分内容为:总里程：总共行驶里程。\n● 能耗曲线\n*画面仅供参考，请以产品实际为准\n动力输出为蓝色，能量回收为绿色。 仪表显示屏中间信息区域\n*画面仅供参考，请以产品实际为准\n表示当前输出动力或能量回收时的瞬时功率与 大功率的百分比：\n● 瞬时功率百分比 0%~80%时，显示为蓝色 （浅色模式）/白色（深色模式）能量条，表 示动力系统输出动力。\n● 瞬时功率百分比在 80%~100%时，显示为 红色能量条，表示动力系统输出动力。\n● 瞬时功率百分比小于 0%时，显示为绿色能 量条，表示制动系统能量回收。\n仪表显示屏中间信息区域根据使用的驾驶辅助 功能显示相关信息。\n仪表显示屏右侧信息区域\n仪表显示屏右侧信息区域显示电话信息、音乐 信息，告警信息等。\n瞬时功率百分比\n22 车辆概览 抬头显示\n开启抬头显示，当您在驾驶时，无需低头，就 能在前风挡玻璃上看到车辆的当前车速、导航 等信息。您还可以打开AR 增强显示，无需佩 戴智能设备，即可看到 3D 特效与真实道路元 素结合的动态实景，帮助您更直观的进行智能 辅助驾驶。\n•抬头显示系统位于驾驶员前方的仪表台处。\n•驾驶前请检查并确认抬头显示影像的位置和 亮度不会妨碍安全驾驶。\n•抬头显示仅支持显示华为智驾的导航信息。\n设置抬头显示\n抬头显示默认开启，您可以通过以下方式调节 抬头显示：\n● 在控制中心中调节抬头显示\n*画面仅供参考，请以产品实际为准\n1. 从中控屏顶部向下滑出控制中心，点击编 辑，进入快捷开关编辑状态，将抬头显示快 捷开关添加到控制中心。\n*画面仅供参考，请以产品实际为准\n2. 点击抬头显示快捷开关，调节抬头显示的亮 度和高度。\n● 在设置中调节抬头显示\n*画面仅供参考，请以产品实际为准', '线路检修规程(济南).pdf的序号\t检查项\n点 号\t检查内容\t检查记录\t\n正常\t不正常\t备注\t\n1\t测量\n部分\t廓形测量设备\t□\t□\t\t\n波磨测量设备\t□\t□\t\n2\t打磨\n部分\t打磨模式参数\t□\t□\t\t\n1号车左侧打磨小车、打磨头\n紧固螺栓及位置\t□\t□\t\n2号车左侧打磨小车、打磨头\n紧固螺栓及位置\t□\t□\t\n1号车右侧打磨小车、打磨头\n紧固螺栓及位置\t□\t□\t\n2号车右侧打磨小车、打磨头\n紧固螺栓及位置\t□\t□\t\n打磨磨石损耗、裂缝\t□\t□\t\n3\t消防\n部分\t灭火器压力指示器\n及失效日期\t□\t□\t\t\n便携式灭火器压力指示器\n及失效日期\t□\t□\t\n防火板固定螺栓\t□\t□\t\n4\t照明\n部分\t车辆照明设备工作良好\t□\t□\t\t\n5\t其他\n部分\t无明显漏水漏油现象\t□\t□\t\t\n水箱容量确定\t□\t□\t\n序号\t故障记录\t故障处理情况\t\n1\t\t\t\n2\t\t\t\n3\t\t\t\n的一个表格为:序号\t检查项\n点 号\t检查内容\t检查记录\t\n正常\t不正常\t备注\t\n1\t测量\n部分\t廓形测量设备\t□\t□\t\t\n波磨测量设备\t□\t□\t\n2\t打磨\n部分\t打磨模式参数\t□\t□\t\t\n1号车左侧打磨小车、打磨头\n紧固螺栓及位置\t□\t□\t\n2号车左侧打磨小车、打磨头\n紧固螺栓及位置\t□\t□\t\n1号车右侧打磨小车、打磨头\n紧固螺栓及位置\t□\t□\t\n2号车右侧打磨小车、打磨头\n紧固螺栓及位置\t□\t□\t\n打磨磨石损耗、裂缝\t□\t□\t\n3\t消防\n部分\t灭火器压力指示器\n及失效日期\t□\t□\t\t\n便携式灭火器压力指示器\n及失效日期\t□\t□\t\n防火板固定螺栓\t□\t□\t\n4\t照明\n部分\t车辆照明设备工作良好\t□\t□\t\t\n5\t其他\n部分\t无明显漏水漏油现象\t□\t□\t\t\n水箱容量确定\t□\t□\t\n序号\t故障记录\t故障处理情况\t\n1\t\t\t\n2\t\t\t\n3\t\t\t\n', '大语言模型.pdf的第五章 模型架构的部分内容为:考第 9.2.1节），并且对于模型性能产生的影响也比较小。一些代表性的大语言模 型，如 PaLM [33]和 StarCoder [96]，已经使用了多查询注意力机制。为了结合多 查询注意力机制的效率与多头注意力机制的性能，研究人员进一步提出了分组查 询注意力机制（Grouped-Query Attention, GQA）[172]。GQA将全部的头划分为若 干组，并且针对同一组内的头共享相同的变换矩阵。这种注意力机制有效地平衡 了效率和性能，被 LLaMA-2模型所使用。图 5.4展示了上述两种注意力查询机制。\n•硬件优化的注意力机制. 除了在算法层面上提升注意力机制的计算效率，还 可以进一步利用硬件设施来优化注意力模块的速度和内存消耗。其中，两个具有 代表性的工作是 FlashAttention [173]与 PagedAttention [174]。相比于传统的注意力 实现方式，FlashAttention通过矩阵分块计算以及减少内存读写次数的方式，提高 注意力分数的计算效率；PagedAttention则针对增量解码阶段，对于 KV缓存进行 分块存储，并优化了计算方式，增大了并行计算度，从而提高了计算效率。对于 这些技术的细节将在第 9.2.2节进行介绍。 𝒚 … …\n相加和归一化\n混合专家层\n专家1 专家2 专家3 专家4 专家1 专家2 专家3 专家4\n相加和归一化\n路由\n路由\n𝒙\n注意力层 𝒙𝟏 “中” 𝒙" “国”\n图 5.5 混合专家模型示意图\n5.2.6 混合专家模型\n如第 2.2节所述，大语言模型能够通过扩展参数规模实现性能的提升。然而， 随着模型参数规模的扩大，计算成本也随之增加。为了解决这一问题，研究人员 在大语言模型中引入了基于稀疏激活的混合专家架构（Mixture-of-Experts, MoE）， 旨在不显著提升计算成本的同时实现对于模型参数的拓展。', '小米SU7用户手册.pdf的1. 车辆设置列表\t2. 具体设置显示区域\t\n连接\t可设置基础连接、手车互联等功能。\t\n显示\t可设置中控屏、抬头显示HUD*、仪表屏等功能。\t\n声音\t可设置主驾头枕扬声器、音量调节、声效调节、提示音等功能。\t\n智能设备\t可设置车载冰箱*等车载设备。\t\n智能语音\t可设置小爱同学唤醒和对话、个性化等功能。\t\n安全与服务\t可设置行车记录、哨兵模式、车内高温保护、车辆服务模式等功能。\t\n系统\t可查看和设置系统版本、车辆名称、应用管理、版本信息、隐私声\n明、开发者选项、恢复出厂设置等功能。\t\n个人中心\t登录个人账号后，可设置用车习惯、云同步等功能。\t\n的一个表格为:1. 车辆设置列表\t2. 具体设置显示区域\t\n连接\t可设置基础连接、手车互联等功能。\t\n显示\t可设置中控屏、抬头显示HUD*、仪表屏等功能。\t\n声音\t可设置主驾头枕扬声器、音量调节、声效调节、提示音等功能。\t\n智能设备\t可设置车载冰箱*等车载设备。\t\n智能语音\t可设置小爱同学唤醒和对话、个性化等功能。\t\n安全与服务\t可设置行车记录、哨兵模式、车内高温保护、车辆服务模式等功能。\t\n系统\t可查看和设置系统版本、车辆名称、应用管理、版本信息、隐私声\n明、开发者选项、恢复出厂设置等功能。\t\n个人中心\t登录个人账号后，可设置用车习惯、云同步等功能。\t\n', '小米SU7用户手册.pdf的前言的部分内容为:仪表屏1. 车辆状态信息\n2. 限速标识\n3. 驾驶模式\n4. 挡位信息\n5. 瞬时功率\n6. 车速信息\n7. 电量续航\n仪表屏亮度调节\n请在中控屏下方控制栏打开设置，进入显示>仪表屏亮度，滑动进行调节仪表屏亮度。 请在中控屏下方控制栏打开设置，进入显示>仪表屏>自动亮度调节，点击开启或关闭该功 能。\n自动亮度调节\n说明:\n• 开启自动亮度调节功能后，车辆将通过前风挡上部的阳光雨量传感器对外部环境的 识别，自动进行调节仪表屏亮度。\n• 请保持前风挡上部阳光雨量传感器区域的清洁和无遮挡，以确保阳光雨量传感器部 件的感知能力更准确有效。\n抬头显示（HUD）*\n抬头显示（HUD）是将驾驶员所需要的信息投射到前风挡玻璃上，使驾驶员无需将视线从道 路上移开，就可以看到导航、限速标识、车速等信息。可以提高驾驶员驾驶车辆的安全性和 舒适性，减少疲劳和注意力分散的情况。1. 电量续航信息\n2. 限速标识\n3. 当前车速\n4. 导航信息\n说明:\n为确保抬头显示效果，请定期清洁前风挡玻璃内外侧和抬头显示仪上的灰尘或污物。\n注意:\n• 抬头显示效果在雨天、雪天、阳光强烈等天气条件下可能显示效果不佳。\n• 部分太阳镜可能会影响对抬头显示（HUD）信息的读取。\n危险:\n请勿在行驶过程中长时间将视线放在抬头显示界面上，以免分散注意力，影响驾驶安 全。\n使用中控屏开启和关闭\n请在中控屏下方控制栏打开设置，进入显示>抬头显示HUD，点击开启或关闭该功能。\n抬头显示主题设置\n请在中控屏下方控制栏打开设置，进入显示>抬头显示HUD>主题，可设置抬头显示的主 题。\n抬头显示各主题显示信息如下：', "Lockheed_martin_F35_user_manual-1 (2).pdf的43 ft (13.1 m) Wing Area 460 ft² (42.7m²) 460 ft² (42.7 m²) 668 ft² (62.1 m²) Empty weight 29,098 lb (13,199kg) 32,300 lb (14,700 kg) 34,800 lb (15,800 kg) Internal fuel 18,498 lb (8,390kg) 13,326 lb (6,045 kg) 19,624 lb (8,900 kg) Max takeoff weight 70,000 lb class (31,800 kg) 60,000 lb class (27,300 kg) 70,000 lb class (31,800 kg)的部分内容为:Six additional passive infrared sensors are distributed over the aircraft as part of Northrop Grumman's electro-optical AN/AAQ-37 Distributed Aperture System (DAS), which acts as a missile warning system, reports missile launch locations, detects and tracks approaching aircraft spherically around the F-35, and replaces traditional night vision devices. All DAS functions are performed simultaneously, in every direction, at all times. The electronic warfare systems are designed by BAE Systems and include Northrop Grumman components. Functions such as the Electro-Optical Targeting System and the electronic warfare system are not usually integrated on fighters. The F-35's DAS is so sensitive, it reportedly detected the launch of an airto-air missile in a training exercise from 1,200 mi (1,900 km) away, which in combat would give away the location of an enemy aircraft even if it had a very low radar cross-section.\nThe communications, navigation and identification (CNI) suite is designed by Northrop Grumman and", 'c1572eb4022820f098b1599178c931ed.pdf的第 11章 编译与下装的部分内容为:11.1.2.2 要求\n要执行增量编译，请按以下步骤操作：\n（1）在“工程”菜单中，选择“编译”命令。 11.1.2.4 结果\n在“语法检查”窗口显示“ADD_COMPILE”编译完成、并报告编译错误和警告信息。 11.1.3编译结果\n编译完成后，会在 AutoThink下方信息窗口显示编译结果，有以下两种可能：\n\uf06e 一是编译全部正确，如下图中(b)所示，这表明工程全部正确。\n\uf06e 二是编译有错误或警告，如下图中(a)所示，这表明工程有部分错误，并用红色字显示错误信 息。双击编译错误可以直接定位到该处错误，进行修改。\n11.1.4检查编译错误\n11.1.4.1 简介\n在信息输出窗口的“语法检查”标签页中，可查看编译是否成功、是否有编译错误信息。如发生 错误，请修改后，重新编译。 已执行编译操作。 11.1.4.3 步骤\n11.1.4.2\n要检查编译错误，请按以下步骤操作：\n（1）在“语法检查”标签页，双击错误信息行，光标自动定位到程序错误位置，对应元素呈选中状 态。\n（2）修改错误。\n（3）重新进行编译。\n11.2 通讯设置\n11.2.1.1 简介\n下装前，需要建立本地计算机和控制器间的通讯连接。你需要分别设置本地计算机和控制器的通 讯参数。\n11.2.1.2 要求\n本地计算机与控制器间通讯线连接正常。 11.2.1.3 步骤\n要设置控制器通讯参数，请按以下步骤操作：', 'SR1-1M-34VIEW.pdf的1-106的部分内容为:Fuze Classification ......................................... 2-1 Methods of Arming ........................................ 2-1 Methods of Functioning ................................. 2-2 Fuze Safety Features ...................................... 2-3 M904 Nose Fuze ........................................... 2-6 M905 Tail Fuze .............................................. 2-7 M907 Nose Fuze ........................................... 2-8 MK 339 Nose Fuze ........................................ 2-9 FMU-26 Fuzes ............................................... 2-13 FMU-54 Tail Fuzes ........................................ 2-16 FMU-56D/B Nose Fuze ................................. 2-18 FMU-72/B Long-Delay Fuze ........................ 2-21 FMU-81/B Short-Delay Fuze ........................ 2-24 FMU-110/B Proximity Nose Fuze ................ 2-26 FMU-113/B Proximity Nose Fuze ................ 2-28 FMU-124A/B Impact Fuze ........................... 2-29 FMU-124B/B Impact Fuze ............................ 2-31 ADU-421A/B Fuze Adapter .......................... 2-31 FMU-139A/B Fuze ........................................ 2-31 FMU-143 Fuze .............................................. 2-33 FZU-1/B Booster ........................................... 2-35 FZU-2/B Booster ........................................... 2-35 FZU-32B/B Initiator ...................................... 2-35 FZU-39/B Proximity Sensor ......................... 2-36 FZU-48/B Initiator ........................................ 2-39 Battery Firing Device (BFD) ......................... 2-42 M147A1 Nose fuze ........................................ 2-42 DSU-33B/B Proximity Sensor ....................... 2-42A MAU-162 Firing Lanyard Adjuster .............. 2-43 Swivel and Link Assembly ............................ 2-43 Retaining Clips .............................................. 2-43 Fuze/Bomb Options ....................................... 2-44', '小米SU7用户手册.pdf的前言的部分内容为:• 小米汽车 APP：手机蓝牙钥匙开通后，打开小米汽车 APP进入车辆来进行解闭锁车辆、 开启和关闭充电口、开启前备箱、启动车辆、闪灯、开启和关闭后备箱、鸣笛。\n• 无感操作：手机蓝牙钥匙开通后，使用车辆外门把手即可解锁并开启车门，如靠近自动 解锁功能和离车自动锁车功能处于开启状态下，携带手机靠近或离开车辆时，车辆将自 动进行解锁或闭锁。\nNFC钥匙：手机NFC钥匙开通后，将手机贴在主驾侧 B柱 NFC钥匙感应区域，即可解闭 锁车辆，解锁后 2min 内进入主驾位置可直接启动车辆。\n钥匙开通\n打开小米汽车 APP进入车辆>钥匙与安全，对车辆钥匙进行开通和管理。\n蓝牙钥匙\n蓝牙钥匙开通需要您进入车内，并保持车辆处于解锁状态且在 P挡下。\n手机蓝牙钥匙开通流程：\n1. 打开小米汽车 APP进入车辆>钥匙与安全>我的钥匙，进入钥匙管理界面。\n2. 找到手机蓝牙钥匙，点击去开通。\n3. 等待钥匙创建中。\n4. 点击配对，进行手机与车辆匹配连接。\n5. 显示创建成功，点击确认，完成手机蓝牙钥匙开通。\n说明:\n• 手机蓝牙钥匙如未创建成功，请您再次进行上述开通流程，如还未开通成功，请您 联系小米汽车服务中心。\n• Android 系统手机蓝牙钥匙开通后，在车辆附近并与车辆建立连接后，下拉手机通 知栏会显示蓝牙的连接状态。如果显示车辆已连接，便可进行车辆解闭锁、开启前 备箱或开启后备箱的操作。\n为了提高手机蓝牙钥匙进行解闭锁车辆的成功率，需要您将手机蓝牙功能始终保持开启状 态，并对小米汽车 APP进行以下授权：\n• Android：授权小米汽车 APP始终允许定位权限、授权小米汽车 APP自启动权限，将省 电策略设置为无限制，并将小米汽车 APP锁定在后台。', 'SR1-1M-34VIEW.pdf的20MM AMMUNITION . . . . . . . . . . . . . . . . . . . . .1-99的部分内容为:SEQUENCE . . . . . . . . . . . . . . . . . . . . . . . . . . . .2-24 FMU-81/B SAFETY FEATURES . . . . . . . . . . . . .2-24 FMU-81/B SHORT-DELAY FUZE . . . . . . . . . . .2-24 FORMATION DECONFLICTION . . . . . . . . . . . . .6-6 FRAGMENTATION . . . . . . . . . . . . . . . . . . . . . . . .1-4 FRAGMENTATION BOMBS . . . . . . . . . . . . . . . 1-2A FUZE ARMING . . . . . . . . . . . . . . . . . . . . . . . . . . .6-2 FUZE CLASSIFICATION . . . . . . . . . . . . . . . . . . .2-1 FUZE SAFETY FEATURES . . . . . . . . . . . . . . . . .2-3 FUZE SUMMARY TABLE . . . . . . . . . . . . . . . . . .8-3 FUZES . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .2-1 FZU-1/B BOOSTER . . . . . . . . . . . . . . . . . . . . . . .2-35 FZU-2/B BOOSTER . . . . . . . . . . . . . . . . . . . . . . .2-35 FZU-32B/B INITIATOR . . . . . . . . . . . . . . . . . . . .2-35 FZU-39/B PROXIMITY SENSOR . . . . . . 1-94F, 2-36 FZU-48/B INITIATOR . . . . . . . . . . . . . . . . . . . . .2-39', "Lockheed_martin_F35_user_manual-1 (2).pdf的43 ft (13.1 m) Wing Area 460 ft² (42.7m²) 460 ft² (42.7 m²) 668 ft² (62.1 m²) Empty weight 29,098 lb (13,199kg) 32,300 lb (14,700 kg) 34,800 lb (15,800 kg) Internal fuel 18,498 lb (8,390kg) 13,326 lb (6,045 kg) 19,624 lb (8,900 kg) Max takeoff weight 70,000 lb class (31,800 kg) 60,000 lb class (27,300 kg) 70,000 lb class (31,800 kg)的部分内容为:The AN/AAQ-37 electro-optical Distributed Aperture System (DAS) is the first of a new generation of sensor systems being fielded on the Lockheed Martin F-35 Lightning II Joint Strike Fighter. DAS consists of six high resolution Infrared sensors mounted around the F-35 airframe in such a way as to provide unobstructed spherical (4π steradian) coverage and functions around the aircraft without any pilot input or aiming required.\nThe DAS provides three basic categories of functions in every direction simultaneously:\nMissile detection and tracking (including launch point detection and countermeasure cueing)\nAircraft detection and tracking (Situational awareness IRST & air-to-air weapons cueing)\nImagery for cockpit displays and pilot night vision (imagery displayed onto the helmet mounted display)\nThe F-35's DAS was flown in military operational exercises in 2011 and has also demonstrated the ability to detect and track ballistic missiles to ranges exceeding 800 miles (1300 kilometers) and has also\ndemonstrated the ability to detect and track multiple small suborbital rockets simultaneously in flight.\nELECTRO-OPTICAL TARGETING SYSTEM (EOTS)", 'Lockheed_martin_F35_user_manual-1 (2).pdf的12 – BARO / CABIN PRESSURIZATION / GCAS / ALOW This area reports the current Barometric setting, cabin pressurization information, Ground Collision Avoidance System (GCAS) status and ALOW (Altitude Low Warning) setting will allow the pilot to set BARO and ALOW values.的部分内容为:This page is accessible from the additional MENU page though the LITES> button. In this page the following functions are selectable:\nCONSL toggles the cockpit console pages\nPOSIT – toggles the wing navigation lights\nSTROB – toggles the strobe lights\nFORM – toggles the formation lightsHELMET MOUNTED DISPLAY SYSTEM\nThe F-35 Helmet Mounted Display System (HMDS) displays biocular video and symbology information on the helmet visor, providing pilots with all information necessary to execute both day and night missions under a single integrated configuration.\nThe flight data are presented to the pilot in a Virtual Head Up Display, that is it appears in front of the pilot like if it were on an exceptionally wide, frameless Head Up display.\nIn the real-world, the system enables pilots to accurately cue onboard weapons and sensors using the helmet display. Finally, the system also provides “Enhanced Reality” features, like night vision and the\npossibility to look “through the aircraft” thanks to the Distributed Aperture System (DAS). When the pilot moves his head, part of the VHUD symbology is kept aligned with the aircraft boresight, while the rest follows the pilot head movement.only the following fixed symology is shown', 'Lockheed_martin_F35_user_manual-1 (2).pdf的43 ft (13.1 m) Wing Area 460 ft² (42.7m²) 460 ft² (42.7 m²) 668 ft² (62.1 m²) Empty weight 29,098 lb (13,199kg) 32,300 lb (14,700 kg) 34,800 lb (15,800 kg) Internal fuel 18,498 lb (8,390kg) 13,326 lb (6,045 kg) 19,624 lb (8,900 kg) Max takeoff weight 70,000 lb class (31,800 kg) 60,000 lb class (27,300 kg) 70,000 lb class (31,800 kg)的部分内容为:F-35 the TSD screen can zoom and pan to specific locations.\n– one of the of the F-35 is its\nthe information coming from a wide variety of sources in order to provide the pilot an unprecedented situational awareness.\nmost advanced fighter in the world today.\nof the Lightning II ever seen\nTHE most\nCOCKPIT\nThe F-35 features a full-panel-width "Panoramic Cockpit Display" (PCD) glass cockpit, with dimensions of 20 by 8 inches (50 by 20 centimeters).\nA cockpit speech-recognition system (Direct Voice Input) provided by Adacel is planned to improve the pilot\'s ability to operate the aircraft over the current-generation interface.\nThe F-35 will be the first US operational fixed-wing aircraft to use this system, although similar systems have been used in AV-8B and trialed in previous US jets, particularly the F-16 VISTA. In development the system has been integrated by Adacel Systems Inc. with the speech recognition module supplied by SRI\nInternational. The pilot flies the aircraft by means of a right-hand side stick and left-hand throttle.', 'Lockheed_martin_F35_user_manual-1 (2).pdf的43 ft (13.1 m) Wing Area 460 ft² (42.7m²) 460 ft² (42.7 m²) 668 ft² (62.1 m²) Empty weight 29,098 lb (13,199kg) 32,300 lb (14,700 kg) 34,800 lb (15,800 kg) Internal fuel 18,498 lb (8,390kg) 13,326 lb (6,045 kg) 19,624 lb (8,900 kg) Max takeoff weight 70,000 lb class (31,800 kg) 60,000 lb class (27,300 kg) 70,000 lb class (31,800 kg)的部分内容为:includes the Multifunction Advanced Data Link (MADL), as one of a half dozen different physical links. The F35 will be the first fighter with sensor fusion that combines radio frequency and IR tracking for continuous alldirection target detection and identification which is shared via MADL to other platforms without\ncompromising low observability. The non-encrypted Link 16 is also included for communication with legacy systems. The F-35 has been designed with synergy between sensors as a specific requirement, the aircraft\'s "senses" being expected to provide a more cohesive picture of the battlespace around it and be available for use in any possible way and combination with one another; for example, the AN/APG-81 multi-mode radar also acts as a part of the electronic warfare system.\nMuch of the F-35\'s software is written in C and C++ due to programmer availability, Ada83 code also is reused from the F-22. The Integrity DO-178B real-time operating system (RTOS) from Green Hills Software runs on COTS Freescale PowerPC processors.\nThe electronic warfare and electro-optical systems are intended to detect and scan aircraft, allowing\nengagement or evasion of a hostile aircraft prior to being detected.', '医药生物行业周报：持续血糖监测国产替代进程有望提速.pdf的图表目录的部分内容为:根据公告，此次合作旨在增加中国低线城市的放射治疗量，对国内引进全球 高端医疗装备、提升智慧放疗水准具有重大的标志性意义。新成立的合资企业将 帮助中国达到国际原子能机构（IAEA）的建议，即每百万人至少拥有 4个放射治 疗单位，每百万人至少 1.5 个单位。\n（资料来源：根据公司公告整理）\n\uf06e美敦力与一影医疗达成战略合作：拓展智能骨科可及性\n近日，美敦力中国骨科与神外业务集团旗下智能设备与神外事业部宣布与一 影医疗达成战略合作，此次合作整合了美敦力的骨科导航系统及一影医疗的移动 式三维 C 型臂，进一步完善了美敦力 SURGICAL SYNERGY™智能骨科一体化产业\n生态圈，以更为多元化的解决方案，为临床术者提供更智能、可及的新疗法，推 动智能骨科疗法的广泛可及。\n（资料来源：根据公司公告整理）\n\uf06e台湾骨王有限公司的（Surglasses）旗下的 Caduceus S.Caduceus S AR 脊柱导航系统完成第一例增强现实（AR）手术\n近日，台湾骨王有限公司的（Surglasses）旗下的 Caduceus S.Caduceus S AR 脊柱导航系统取得了里程碑式的成功，该系统在美国首次使用 2D C 臂成像进 行增强现实手术。\n该 AR 导航允许外科医生，在手术过程中像透视一样对患者进行更安全的脊 椎手术视野。配备了多个影像追踪器，并通过 3D AR 头戴式显示实时病患体内结 构。且 Caduceus S 配备了穿透式的头戴式显示器，并具备了手术导引的功能， 外科医师能够在手术中同时拥有完整的真实世界手术视野和扩增实境影像导引。 使外科医生能够专注于患者，而不需要频繁的抬头、转头观看手术区域外的显示 系统。并具备多种优势使其外科医生用得其手。', 'SR1-1M-34VIEW.pdf的2-17的部分内容为:The fuze must sense passing through the outer gate (HOB plus 500 feet) and the middle gate (HOB plus 250 feet) after 3.7 seconds and prior to the HOB. Because of a receding target (i.e., the release aircraft), the range gate logic prevents the fuze from functioning. The fuze senses and remembers the range gates if they occur after 3.7 seconds and prior to or after arming time, because the fuze radar is operational at 3.7 seconds, regardless of the set arming time. If all other criteria are satisfied and the munition passes through both range gates and the HOB before arming, the munition functions immediately upon expiration of the arming time.\nRipple releases of up to 12 FMU-56D/B fuzed munitions may be accomplished. However, if more than six munitions are rippled on one pass, an increased dud rate must be expected. For all FMU-56D/B ripple releases, the munitions must attain a spatial separation of 20 feet when HOB is 2200 feet or lower, 24 feet when HOB is 2500 feet, and 38 feet when HOB is 3000 feet. Refer to TO SR1F-15SF-34-1-3 or TO 1-1M44FD to determine minimum release altitude and minimum release interval settings which provide adequate munition separation distance for ripple release.\nThe FMU-56D/B fuze (figure 2-14) is a self -powered doppler radar proximity nose fuze used to open SUU30 dispensers. It has 10 arming time settings and 10 HOB settings (figure 2-15) for above ground fuze functioning. Additionally, the FMU-56 fuze has provisions for selecting an Electronic Countermeasures (ECM) mode. Refer to the Fuze/Bomb Summary Chart (figure 2-37) for details.\nFMU-56D/B ARMING AND OPERATING SEQUENCE', 'c1572eb4022820f098b1599178c931ed.pdf的第 11章 编译与下装的部分内容为:（1）工具栏单击“下装”按钮，将弹出工程比较对话框。\n（2）单击“是”，将弹出登录控制器对话框。\n（3）输入控制器账号和密码，登录控制器，此时，将上传控制器用户工程文件。信息输出窗口显示“上 传用户工程完成”，并弹出工程账户对话框。（4）输入工程账户和密码，将显示工程比较对话框。\n（4）输入工程账户和密码，将显示工程比较对话框。（5）勾选需要比较的工程节点，单击“比较”按钮，将显示两个工程的差异节点和比较结果。\n（5）勾选需要比较的工程节点，单击“比较”按钮，将显示两个工程的差异节点和比较结果。（6）选中不相同的节点，单击“详细比较”，将列出详细的差异内容。\n（6）选中不相同的节点，单击“详细比较”，将列出详细的差异内容。（7）确认工程无误后，关闭工程比较对话框，继续下装。 即初始化下装，下装全部组态逻辑信息和硬件配置信息到控制器，下装时，对应控制器停止运 算、输出，当出现以下情况（中的一种）时，将进行初始化下装操作：\n（7）确认工程无误后，关闭工程比较对话框，继续下装。 即初始化下装，下装全部组态逻辑信息和硬件配置信息到控制器，下装时，对应控制器停止运 算、输出，当出现以下情况（中的一种）时，将进行初始化下装操作：11.3.2全下装\n11.3.2.1 简介\n\uf06e 工程初次下装。\n\uf06e 当前工程与控制器工程不一致或版本不连续。\n\uf06e 执行全编译操作后。 11.3.2.2 要求\n工程已编译通过。 11.3.2.3 步骤', '问界m9-product-manual.pdf的前言 3 目录的部分内容为:车外后视镜 ( 86 页) 后风挡雨刮 ( 100 页) 后视摄像头\n全景环视摄像头 ( 171 页)\n充电口盖 ( 260 页)\n10 11 12\n车辆概览 13 内部简介\n通过下图，您可以了解车辆内部的常用部件。\n前排常用部件\n车窗按键 ( 73 页)\n四门解闭锁按键\n驾驶员监测摄像头\n组合控制拨杆 ( 92 页)\n仪表显示屏 ( 16 页)\nHUD 投影仪区域\n换挡控制拨杆 ( 114 页) 阅读灯开关 ( 98 页)\nSOS 报警按键 ( 305 页)\n内后视镜 车内摄像头 高音扬声器\n遮阳板 ( 109 页)\n14 车辆概览\n空调出风口 高音扬声器 中音扬声器\n危险警告灯按键 ( 95 页) 无线充电仓\n智慧水晶旋钮 ( 121 页) 杯托 ( 104 页)\n加速踏板\n制动踏板\n方向盘按键（右） ( 84 页) 喇叭开关 ( 84 页)\n方向盘按键（左） ( 84 页) 车门开关 ( 65 页) 后排常用部件\nMagLinkTM 接口\n后控制面板 ( 39 页) 老板按键 ( 76 页)\nType-C 接口 ( 102 页)\n冷暖箱 ( 107 页)\n前排座椅靠背储物板 ( 106 页)\n10 11 12\n车辆概览 15 仪表显示屏\n仪表显示屏简介\n您在使用车辆时，仪表显示屏会显示车辆的运行参数及车辆状态，请您务必认真阅读这部分内容，其 中仪表指示灯 ( 17 页) 的信息尤为重要。', '小米SU7用户手册.pdf的前言的部分内容为:• 智驾学堂需要您在车辆上登录个人账号并完成智能辅助驾驶功能授权后开启。\n• 智驾学堂中学习并通过考试的智能辅助驾驶功能仅针对当前登录的账号，如车辆更 换登录账号，该登录账号需学习并通过考试后才能开启相应智能辅助驾驶功能。\n• 仅在车辆处于 P挡下才能使用中控屏进入互动式教学与考试。\n• 您还可以通过手机在线浏览教学视频。\n注意:\n• 智驾学堂必修科目中的警示内容对驾驶安全有重要影响。\n• 如果您将车辆授权给他人使用，为安全起见，请确保被授权用户先登录其个人账号 完成智驾学堂学习并通过考试后，再使用智能辅助驾驶功能。\n• 如果车辆更换驾驶员，为安全起见，请新驾驶员进入智驾学堂进行学习，未经过学 习并通过考试的驾驶员应避免使用智能辅助驾驶功能。\n• 智驾学堂的课程及考试要求可能会随着智能辅助驾驶功能的变化和法规要求而调 整。 路面感知系统是一套可视化模拟显示现实环境、呈现智驾系统信息的视觉交互系统。该功能 默认开启，可以在中控屏上模拟显示车辆周围的交通环境。\n路面感知系统\n注意:\n• 路面感知系统对各种物体、车辆、骑行者或行人可能存在着错误的显示，并非在任 何情况下，都能准确显示周围环境的全部状况，驾驶员应时刻关注交通、道路和车 辆情况，安全驾驶。\n• 由于技术的发展、车辆软硬件及外部环境的复杂性，以下并未尽述路面感知系统功 能无法正常工作或者受到抑制的情形。路面感知系统功能仅用于辅助驾驶员驾驶， 驾驶员应时刻关注交通、道路和车辆情况，安全驾驶。\n路面感知系统功能在以下条件可能无法正常工作，包括但不仅限于：\n• 摄像头受下雨、大雾、沙尘暴和冰雪等恶劣天气影响时，可能导致摄像头性能下降。\n• 摄像头安装位置被篡改。\n• 摄像头被遮挡或脏污。\n• 摄像头失焦或故障。', 'c1572eb4022820f098b1599178c931ed.pdf的第 9章 硬件组态的部分内容为:9.1.2 组态自由通信协议\n9.1.2.1 简介\n当 LX PLC与其它设备进行串口通讯时，若这些第三方设备所支持的协议不是标准的 MODBUS RTU协议，则可以通过自由协议组态来实现。\n9.1.2.2 要求\n编译通过 9.1.2.3 步骤\n当 LX PLC与其它设备进行串口通讯时，若这些第三方设备所支持的协议不是标准的 MODBUS RTU协议，则可以通过自由协议组态来实现。\n举例说明：将 LX PLC作为 MODBUS RTU主站，介绍自由通讯的组态方法。\n（1）设置主站通讯参数\n先按照另一端通讯设备的通讯参数设置 LX控制器的通讯参数，详见设置主站通讯参数。\n假设本例程中 LX PLC通过圆形接口与另一 MODBUS从站设备通讯，该设备从站地址为 02。LX PLC周期读取该从站通讯设备的一个字（WORD），其寄存器地址为 3150，于是根据 MODBUS通讯协议，就能够得到主站需要发送的询问帧为（16进制）：02 04 0C 4D 00 01 A2 BE。\n（2）将数据赋值给存储区\n将主站发送的数据帧中的数据顺序赋值给 PLC中一段连续中间存储区，如图所示，此处 16# 开头的常量表示该常量是 16进制表示方式。本实例将数据存放在从%MB100开始的 8个字节中， 在实际使用中存储区的地址可根据需要自行选择。\n（3）发送数据将询问帧的数据通过串口发送出去。此处需要调用功能块“COMM_SEND（自由协议通讯数据 发送）”，如图所示。\n为实现周期读取从站寄存器的数据，需要调用一个“BLINK”脉冲信号发生器，详见调用 BLINK 功能块。', '大语言模型.pdf的第十二章 评测的部分内容为:• MATH数据集. MATH数据集 [19]包含了 12,500条具有挑战性的数学竞赛 问题。这些问题覆盖了众多的数学领域与知识点，从而确保了数据集的多样性和 难度。每条问题都配备了详细的解题过程，这些过程为模型提供了解决问题的详 细步骤。在MATH数据集中，每个问题都有一个 1到 5之间的难度标注，数字越大 表示问题的难度越高，需要更复杂的数学知识和推理能力才能解决。此外，MATH 数据集中的问题描述和答案均采用 LaTeX格式进行表达。在评估过程中，研究人 员采用答案准确率作为主要评测指标，通过对比模型输出的答案表达式与参考表 达式的等价性来判断答案的正确性。\n问题：There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the workers plant today? 解答： There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 15 = 6. The answer is 6.\n问题：If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n解答： There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n问题：Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n解答： Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 +\n42 = 74. After eating 35, they had 74 35 = 39. The answer is 39.\n例 12.7 数学推理任务 GSM8K示例\n主要问题\n尽管大语言模型在解决复杂推理任务方面已经取得了显著进展，但其仍然存 在一些重要的局限性。', '小米SU7用户手册.pdf的前言的部分内容为:• 标准：抬头显示标准主题中显示电量续航、地图导航、车速、限速等信息。\n• 简洁：抬头显示简洁主题中显示车速、限速、简要导航信息。\n抬头显示调节\n1. 请在中控屏下方控制栏打开设置，进入显示>HUD显示调节。2. 使用方向盘右侧控制区的左按键或右按键切换调节的亮度、高度、横向或角度。\n2. 使用方向盘右侧控制区的左按键或右按键切换调节的亮度、高度、横向或角度。3. 上下滚动方向盘右侧的滚轮按键进行调节数值。 抬头显示亮度会通过车辆前风挡上部阳光雨量传感器对外部环境的识别，车辆会自动进 行调节。\n危险:\n请勿在车辆行驶过程中频繁调节抬头显示。为保证驾驶安全，请在道路和交通条件允许 时，进行抬头显示的调节。\n空调控制', "Lockheed_martin_F35_user_manual-1 (2).pdf的43 ft (13.1 m) Wing Area 460 ft² (42.7m²) 460 ft² (42.7 m²) 668 ft² (62.1 m²) Empty weight 29,098 lb (13,199kg) 32,300 lb (14,700 kg) 34,800 lb (15,800 kg) Internal fuel 18,498 lb (8,390kg) 13,326 lb (6,045 kg) 19,624 lb (8,900 kg) Max takeoff weight 70,000 lb class (31,800 kg) 60,000 lb class (27,300 kg) 70,000 lb class (31,800 kg)的部分内容为:The Electro-Optical Targeting System (EOTS) for the F-35 Lightning II is an affordable, high-performance, lightweight, multi-function system that provides precision air-to-air and air-to-surface targeting capability. The low-drag, stealthy EOTS is integrated into the F-35 Lightning II's fuselage with a durable sapphire window and is linked to the aircraft's integrated central computer through a high-speed fiber-optic interface.\nAdvanced EOTS, an evolutionary electro-optical targeting system, is available for the F-35’s Block 4\ndevelopment. Designed to replace EOTS, Advanced EOTS incorporates a wide range of enhancements and upgrades, including short-wave infrared, high-definition television, an infrared marker and improved image detector resolution.SENSOR FUSION\nA key feature of 5th generation fighters is sensor/information fusion: information coming from different\nsubsystems (Radar, Radar Warning Systems, EOTS, DAS and datalink) is collected, compared and\nintegrated in a single intuitive battlespace depiction in the Tactical Situation Display, providing the pilot with unparalled situational awareness. The F-35 can also easily share this information with other assets.FLIGHT CONTROL SYSTEM", '问界m9-product-manual.pdf的第二排座椅调节的部分内容为:交通信号灯识别（2D 显示） 11\nCity LCC Plus 支持 2D 显示当前车道的交通 信号灯，仅订阅 ADS 高阶包后可用。\nCity LCC Plus 利用地图和摄像头获得当前车 道的标准机动车交通信号灯指示信息，并在仪 表显示屏上 2D 显示。可识别的机动车交通信 号灯包含球形灯、箭头灯和倒计时灯，可识别\n驾驶辅助 149 的信息包括信号灯的颜色、箭头方向（如有）\n12\n•路口交通情况复杂，请驾驶员务必时刻关注 仪表显示屏提示、声音提示和周围环境，必 要时及时接管，确保安全驾驶。\n•City LCC Plus 识别的机动车交通信号灯 信息并非始终准确，切勿过度依赖 City LCC Plus 识别的机动车交通信号灯信息驾 驶。机动车交通信号灯位置变化、数量增加 或减少、发生故障等原因，均可能导致 City LCC Plus 识别错误。\n•驾驶员应始终保持警惕，密切注意周围各种 危险情形，必要时及时人工干预或接管车辆 （例如适当减速、制动、转向等），确保安 全驾驶。违反上述操作会影响您的安全驾 驶，可能会引发事故，甚至导致财产损毁、 人身伤亡。\n路口通行\nLCC 可以一定程度地辅助驾驶员控制车辆通过 标准直行路口和分叉路口，通过路口的具体能 力取决于车辆是否订阅了ADS 高阶包。\n● 标准路口直行（City LCC）\n若未订阅 ADS 高阶包，则车辆仅能使用 City LCC，无法使用 City LCC Plus。', '问界m9-product-manual.pdf的前言 3 目录的部分内容为:1. 在中控屏进入设置 ＞显示＞ 高级显示设 置。\n*画面仅供参考，请以产品实际为准 10\n2. 打开抬头显示开关，点击进入设置界面，调 节抬头显示的亮度和高度。\n11\n车辆概览 23\n12\n3. 在 ARHUD 抬头显示卡片左侧区域向上滑\n您可以通过同步调节座椅高度和抬头显示的高\n请使用清洁干燥的微纤维布轻轻擦拭抬头显示 投影仪区域，如有不易擦净的污渍，可将微纤\n•请勿在抬头显示投影仪或风挡玻璃投影区域 上放置任何物品和贴纸，否则可能会中断抬 头显示指示。\n•请勿触摸抬头显示投影仪内部或向投影仪内 部投掷边缘尖锐的物体，否则可能会导致机\n•请勿在抬头显示投影仪附近放置任何盛有液 体的容器，如果液体进入投影仪区域，可能 会导致电气故障。\n24 车辆概览 多屏联动\n若您的车辆已选配后排娱乐屏，您可将中控 屏、副驾屏和后排屏联动起来，全车乘客一起\n*画面仅供参考，请以产品实际为准\n2. 点击中控屏或副驾屏状态栏中的 图标，进 入多屏管理界面。\n3. 长按界面中正在播放的视频，拖拽到您希望 一起观看的屏幕中，目标屏将同步播放视频 内容。拖拽到全车共享按键 ，全车娱乐屏 将同步播放视频内容。\n4. 在任一同看屏幕的视频窗口进行播放控制， 如暂停、调整进度、切换视频等，所有同看 屏幕的视频播放将同步变化。\n退出多屏同看\n1. 点击中控屏或副驾屏状态栏中的 图标，进2. 点击目标屏图标中的结束分享按键 ，该屏\n•为保证行车安全，驾驶员请勿在驾驶过程中 操作及设置中控屏，如需操作请驻车并确保', 'SR1-1M-34VIEW.pdf的20MM AMMUNITION . . . . . . . . . . . . . . . . . . . . .1-99的部分内容为:FINS SUBSYSTEM . . . . . . . . . . . . . . . . . . . . . . . .3-1 FIRE FIGHTING AND EVACUATION . . . . . . .1-105 FIRE FIGHTING AND EVACUATION, BOMBS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .1-105 FIRE FIGHTING AND EVACUATION, CBUs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .1-105 FIRE FIGHTING AND EVACUATION, MISSILES . . . . . . . . . . . . . . . . . . . . . . . . . . . .1-105 FLIR LOS OPTIONS . . . . . . . . . . . . . . . . . . . . . . .3-5 FMU SERIES FUZING . . . . . . . . . . . . . . . . . . . . . .1-6 FMU-110/B ARMING AND OPERATING\nSEQUENCE . . . . . . . . . . . . . . . . . . . . . . . . . . . .2-26 FMU-110/B ECM MODE . . . . . . . . . . . . . . . . . . .2-27 FMU-110/B PROXIMITY NOSE FUZE . . . . . . .2-26 FMU-110/B SAFETY FEATURES . . . . . . . . . . . .2-27 FMU-113/B ARMING AND OPERATING']

item = {'query': 'F35 的头盔显示','hits':[{'content':i} for i in docs]}


new_item = sliding_windows(item, rank_end=27, window_size=5, step=5)
print(new_item)
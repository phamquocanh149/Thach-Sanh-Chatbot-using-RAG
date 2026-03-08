import json
import random
import re

final_data = []

def fix_json(text):
    text = re.sub(r'//.*', '', text) # Xóa comment
    
    objects = []
    brace_level = 0
    start_idx = -1
    in_string = False
    escape_char = False
    
    for i, char in enumerate(text):
        if char == '"' and not escape_char:
            in_string = not in_string
            
        if not in_string:
            if char == '{':
                if brace_level == 0:
                    start_idx = i
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and start_idx != -1:
                    json_str = text[start_idx:i+1]
                    try:
                        obj = json.loads(json_str)
                        if "messages" in obj:
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass 
                    start_idx = -1
                    
        if char == '\\' and in_string:
            escape_char = not escape_char
        else:
            escape_char = False
            
    return objects

# 1. ĐỌC FILE CÓ CONTEXT
try:
    with open('ques_ans_wcontext.jsonl', 'r', encoding='utf-8') as f:
        raw_content = f.read()
        data_with_context = fix_json(raw_content)
        final_data.extend(data_with_context)
        print(f" Đã nội soi và cứu hộ được {len(data_with_context)} câu CÓ context!")
except Exception as e:
    print(f" Lỗi đọc file có context: {e}")

# 2. ĐỌC FILE KHÔNG CONTEXT
try:
    count_no_context = 0
    with open('thachsanh.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line.strip())
                system_msg = {
                    "role": "system", 
                    "content": "Bạn là Thạch Sanh trong truyện cổ tích Việt Nam. Hãy trò chuyện thân thiện với người dùng và xưng 'tôi'."
                }
                if "messages" in item:
                    item["messages"].insert(0, system_msg)
                    final_data.append(item)
                    count_no_context += 1
            except:
                pass
    print(f" Đã đọc thành công {count_no_context} câu KHÔNG context!")
except Exception as e:
    print(f" Lỗi đọc file không context: {e}")

# 3. TRỘN ĐỀU VÀ XUẤT FILE ĐẦU RA
random.shuffle(final_data)

with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"🎉 HOÀN TẤT! Đã trộn đều và lưu tổng cộng {len(final_data)} cặp hội thoại vào 'train.jsonl'")
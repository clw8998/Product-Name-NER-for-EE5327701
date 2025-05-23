# Product-Name-NER-for-EE5327701

[中文](./README.md)

This project contains a **Traditional Chinese** Named Entity Recognition (NER) model that can be used to extract **various attributes** from **product names**, such as brand, name, color, etc.

## Table of Contents

1. [Installation](#installation)
2. [Usage & Inference Examples](#usage--inference-examples)
3. [Attribute Annotation Guidelines and Accuracy](#attribute-annotation-guidelines-and-accuracy)

## 1. Installation

### 1.1 PyTorch Installation

It is recommended to install from [PyTorch Official Website](https://pytorch.org/get-started/locally/).

### 1.2 Install dependencies

```bash
pip install transformers pandas seqeval
```

## 2. Usage & Inference Examples

Inference examples are provided in the `inference_example.ipynb` notebook. The main adjustable parameters include the input product name and the NER attributes you want to extract.  
You can input the product names and the attributes you want to extract and run the following code to perform inference:

```python
# put attribute here!
all_attribute = ['品牌', '名稱', '產品', '產品序號', '顏色', '材質', '對象與族群', '適用物體、事件與場所', 
                     '特殊主題', '形狀', '圖案', '尺寸', '重量', '容量', '包裝組合', '功能與規格']

# put infernce data here!
inference_data = ['【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊', '【Jabra】Elite 4 ANC真無線降噪藍牙耳機 (藍牙5.2雙設備連接)']

# set device
config.string_device =  'cuda' if torch.cuda.is_available() else 'cpu'
config.device = torch.device(config.string_device)

# load model
model, tokenizer = inference_api.load_model("clw8998/Product-Name-NER-model", device=config.device)

ner_tags = inference_api.get_ner_tags(model, tokenizer, inference_data, all_attribute)
```

### 2.1 Expected Output

The inference result will include structured attribute data with corresponding confidence scores. An example is shown below:

(The confidence score is a numerical representation of how confident the model is in its prediction, with values ranging from 1 to 0, where 1 represents very high confidence, and 0 represents no confidence.)

```json
{
  "【a‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊": {
    "品牌": [["a‵bella", 0.9987], ["浪漫晶飾", 0.9861]],
    "名稱": [["密碼", 0.6318]],
    "顏色": [["深海藍", 0.9486]],
    "材質": [["水晶", 0.9143]]
  }
  ...
}
```

### 2.2 Retrieve Entities for a Specific Product Name

Use the product name to retrieve corresponding entities **(If the product name contains English letters, use lowercase)**, for example:

```python
print('【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊')
result_dict['【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊'.lower()]
```

Result:

```
【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊

{'品牌': [['a‵bella', 0.9987], ['浪漫晶飾', 0.9861]],
 '名稱': [['密碼', 0.6318]],
 '產品': [['手鍊', 0.9991]],
 '產品序號': [],
 '顏色': [['深海藍', 0.9486]],
 '材質': [['水晶', 0.9143]],
 '對象與族群': [],
 '適用物體、事件與場所': [['手', 0.9968]],
 '特殊主題': [],
 '形狀': [['方形', 0.9974]],
 '圖案': [],
 '尺寸': [],
 '重量': [],
 '容量': [],
 '包裝組合': [],
 '功能與規格': []}
```

### 2.3 Retrieve Specific Attribute Entities

If you want to retrieve entities for a specific attribute, for example, brand:

```python
print('【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊')
result_dict['【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊'.lower()]['品牌']
```

Result:

```
【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊

[['a‵bella', 0.9987], ['浪漫晶飾', 0.9861]]
```

## 3. Attribute Annotation Guidelines and Accuracy

| No.  | Attribute (English)           | Description                                                                                     | F1-score |
|------|-------------------------------|-------------------------------------------------------------------------------------------------|----------|
| 1    | **品牌 (brand)**               | The product's brand name, such as Asus, LG                                                      | 0.8770   |
| 2    | **名稱 (name)**                | The "product series" or name created by the seller for special purposes. Excludes special topics or product types. Does not include advertisements. For example, iPhone 12, ROG 3060Ti | 0.5941   |
| 3    | **產品 (product)**             | The actual product name (type), such as computer, mouse, keyboard, monitor, toy, snack, tissue   | 0.8001   |
| 4    | **產品序號 (product serial number)** | The product's unique alphanumeric code, excluding series names                                  | 0.8443   |
| 5    | **顏色 (color)**               | The color information, including cosmetics and brightness. For example, floral red, blue, crystal | 0.8898   |
| 6    | **材質 (material)**            | The material used in the product, not ingredients. For example, wood, PVC, 304 stainless steel  | 0.7810   |
| 7    | **對象與族群 (target group)**   | The target audience, such as newborns, pets, seniors                                            | 0.8755   |
| 8    | **適用物體、事件與場所 (applicable item, event, or place)** | The item, event, or place the product is applicable for. For example, hand-use, cycling, kitchen | 0.7614   |
| 9    | **特殊主題 (special topic)**    | Special themes or brands. For example, One Piece, J.K. Rowling                                  | 0.4979   |
| 10   | **形狀 (shape)**               | The product's shape and appearance description. For example, round, button shape, bear shape     | 0.7793   |
| 11   | **圖案 (pattern)**             | Patterns on the product, such as round or spiral patterns                                       | 0.5731   |
| 12   | **尺寸 (size)**                | Product size, often in numerical or specific form. For example, 120x80x10cm (LxWxH), XL, ATX    | 0.8420   |
| 13   | **重量 (weight)**              | Product weight, often in numerical or specific form. For example, 10g, ultra-light               | 0.9105   |
| 14   | **容量 (capacity)**            | Product capacity, often in numerical or specific form. For example, 128GB (computer), large capacity | 0.9220   |
| 15   | **包裝組合 (package combination)** | The product's packaging method or combination. For example, 10 pieces, 10g/package, keyboard & mouse combo, battery included | 0.7478   |
| 16   | **功能與規格 (function & specifications)** | The product's functions and special specifications. For example, USB3.0, sun protection, waterproof | 0.7960   |
| 17   | **Macro F1-score**             | Overall F1-score                                                                                | 0.7807   |

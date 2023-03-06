### cfg后缀文件
为spacy模型配置文件，sm后缀代表small模型，trf后缀代表transformer大模型。


### get complete cfg file
```angular2html
python -m spacy init fill-config ./base_config_trf.cfg ./config_trf.cfg
```

### 安装en_core_web_trf
```
python -m spacy download en_core_web_trf
```

### train command(sm model):
```
python -m spacy train config_sm.cfg --output ./ --paths.train ./train.spacy --paths.dev ./val.spacy
```

### train command(trf model):
```
python -m spacy train config_trf.cfg --output ./ --paths.train ./train.spacy --paths.dev ./val.spacy
```
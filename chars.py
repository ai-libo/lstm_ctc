# -*- encoding: utf-8 -*-

import re

# common character sets

digits = u"0123456789"
letters = u"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
symbols = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
ascii = digits+letters+symbols
greek = u"ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
chinese = u"住院号年龄性别样本检验目的项代测定值单位参考日期男术前四抗梅毒螺旋体特异弱阳阴丙型肝炎乙表面原人免疫缺陷病粪便常规寄生虫卵镜计数颜色黄状软蛔未查见钩鞭吞噬细胞油滴夏雷登氏脓霉菌孢子红白不消化食物隐血试脑脊液房有核清晰度透明外观无潘肌钙蛋量尿亚硝酸盐质酮浊胆素葡萄糖管类酵母粘丝结晶比重上皮电导率淡脂酶正肿瘤标志铁甲胎癌胚链三系心肾解极低密固醇甘酯机磷高球腺苷脱氨氯钠天门冬基转移总岩藻直接间钾汁氮碱谷酰脯二肽酐反应谱激同工活羟丁氢乳五分嗜粒小板压积淋巴平均布宽中浓含凝功能聚国际准时对照秒纤维部超敏快速法胱抑滤过暗气析全实剩余吸入氧流碳根温纠后离隙饱和渗输棕于限镁套碘游促说女自费褐沉降动脉淀粉引急诊环瓜风湿因溶着点双列黑糊烂绿泳网织百纳端利胸水李凡他块浑或草群传研鳞相关神经元烯角片段枝杆静餐腹补随微视合内科两周颗力旁拷贝载半紧张浆饮立卧醛普通稀建议复成胺一空巨辩果详胰岛释放亮绒毛膜线组增殖纺锤波形尔髓备注呕吐绝底及戊庚非轻澈药克莫司骨谢胶殊序延长会感染荧光多室辅助制个共读植用硬羧集露真干扰严残留灶式图文报告涂其学灰腔仅供糜米可浅肠道种优势歧酪梭拉柔嫩论呼钟帕深祥隆孕泡泌怀雌排刺身滑睾少万鉴请簇所幼稚偶带培养叶层混已橙地辛岁以栓弹发鲎例酚连铜蓝肺支情石份杂交幽现症服附纯疱疹弓行出热综停华林儿茶去肪唾洁季节新咖啡效价器官库融临床澄古峰指受今早抽昨晚则筛耐下大黏每切迹做除次始介坏死灵芝界苯加突变衣军团副与沟"
default = ascii+chinese
european = default+greek

# List of regular expressions for normalizing Unicode text.
# Cleans up common homographs. This is mostly used for
# training text.

# Note that the replacement of pretty much all quotes with
# ASCII straight quotes and commas requires some
# postprocessing to figure out which of those symbols
# represent typographic quotes. See `requote`

# TODO: We may want to try to preserve more shape; unfortunately,
# there are lots of inconsistencies between fonts. Generally,
# there seems to be left vs right leaning, and top-heavy vs bottom-heavy

replacements = [
    (u'[_~#]',u"~"), # OCR control characters
    (u'"',u"''"), # typewriter double quote
    (u"`",u"'"), # grave accent
    (u'[“”]',u"''"), # fancy quotes
    (u"´",u"'"), # acute accent
    (u"[‘’]",u"'"), # left single quotation mark
    (u"[“”]",u"''"), # right double quotation mark
    (u"“",u"''"), # German quotes
    (u"„",u",,"), # German quotes
    (u"…",u"..."), # ellipsis
    (u"′",u"'"), # prime
    (u"″",u"''"), # double prime
    (u"‴",u"'''"), # triple prime
    (u"〃",u"''"), # ditto mark
    (u"µ",u"μ"), # replace micro unit with greek character
    (u"[–—]",u"-"), # variant length hyphens
    (u"ﬂ",u"fl"), # expand Unicode ligatures
    (u"ﬁ",u"fi"),
    (u"ﬀ",u"ff"),
    (u"ﬃ",u"ffi"),
    (u"ﬄ",u"ffl"),
]

def requote(s):
    s = unicode(s)
    s = re.sub(r"''",u'"',s)
    return s

def requote_fancy(s,germanic=0):
    s = unicode(s)
    if germanic:
        # germanic quoting style reverses the shapes
        # straight double quotes
        s = re.sub(r"\s+''",u"”",s)
        s = re.sub(u"''\s+",u"“",s)
        s = re.sub(r"\s+,,",u"„",s)
        # straight single quotes
        s = re.sub(r"\s+'",u"’",s)
        s = re.sub(r"'\s+",u"‘",s)
        s = re.sub(r"\s+,",u"‚",s)
    else:
        # straight double quotes
        s = re.sub(r"\s+''",u"“",s)
        s = re.sub(r"''\s+",u"”",s)
        s = re.sub(r"\s+,,",u"„",s)
        # straight single quotes
        s = re.sub(r"\s+'",u"‘",s)
        s = re.sub(r"'\s+",u"’",s)
        s = re.sub(r"\s+,",u"‚",s)
    return s

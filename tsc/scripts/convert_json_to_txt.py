#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers import convert_json_to_txt

if __name__ == "__main__":
    json_path = 'YOUR_JSON_PATH'
    txt_path = 'YOUR_TXT_FOLDER_PATH'
    data = convert_json_to_txt(json_path, txt_path)

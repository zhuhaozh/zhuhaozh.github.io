#########################################################################
# File Name: post_cmd.sh
# Author: zhuhao
# Created Time: 2018年04月27日 星期五 00时03分58秒
#########################################################################
#!/bin/bash
git add .&&git commit -m 'post'&& jekyll build&&git push 

#########################################################################
# File Name: post_cmd.sh
# Author: zhuhao
# Created Time: 2018年04月27日 星期五 00时03分58秒
#########################################################################
#!/bin/bash
jekyll build && git add .&&git commit -m 'post'&&git push 

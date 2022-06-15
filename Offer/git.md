具体过程：

$ ssh-keygen

$ cat /home/hider/.ssh/id_rsa.pub

//windows--》 clip < ~/.ssh/id_rsa.pub

加入到github的ssh-key中

git clone git@github.com:liuao520/XXXXX.git

移动到本地远程文件夹（禁止有空文件）

$ git status

$ git add .

$ git commit -m "all in"

$ git config --global user.email "1147958934@qq.com"

$ git config --global user.name "liuao520"

$ git push origin master


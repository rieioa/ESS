재활용 레포


git clone https://github.com/rieioa/ESS
git pull origin main --allow-unrelated-histories

작업 시 

git config --global push.autoSetupRemote true
git checkout -b branch_name
git add .
git commit -m "message"
git push 

(안되면 git push --set-upstream origin branch_name)
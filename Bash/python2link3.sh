#How to link  python2.7.17 to python3


# A simple safe way would be to use an alias. Place this into ~/.bashrc or ~/.bash_aliases file:

alias python=python3

#After adding the above in the file, run source ~/.bashrc or source ~/.bash_aliases.
source ~/.bashrc

python --version

# If you want to back python2, you can use

unalias python

pyhton --version


# chmod +rwx  give permission   chmod 750 
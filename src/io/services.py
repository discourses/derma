import subprocess


class Services:

    def __init__(self):

        self.name = 'Services'

    @staticmethod
    def awscli(command):

        push = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

        return push.returncode



import os
import git
folder = os.path.dirname(__file__)
print(folder)
repo = git.Repo()
remote = repo.remote()
remote.pull()
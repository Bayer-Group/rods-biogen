# Debugging template

This document contains a compendium of common problems related to the topic, and how to solve them. **For example**:

## Error when loading file

* **Stacktrace**

```bash
----> 1 with open("non_existent_file", "rb") as f:
      2     f.read()
      3 

FileNotFoundError: [Errno 2] No such file or directory:
```

* **Why it's happening**

This error is caused by trying to open a file that cannot be found.

* **How to solve it**

    Make sure that the path to your target file is correct, and the file already exists.

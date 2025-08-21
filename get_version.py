
def get_version() -> str:
    return "1.22"

def get_lib_version() -> str:
    return "1.22.1"

def get_dependency_string() -> str:
    return "pyort_lib~=1.22.0"

if __name__ == "__main__":
    print(get_version())

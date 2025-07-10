import unittest
import os
import subprocess
import sys



class TestExtensionImport(unittest.TestCase):

    def get_extensions_list(self):
        """
        This method reads setup.py and gets the list of extensions from the setup.py file
        """

        #find the absolute path of this file
        current_file_path = os.path.abspath(__file__)

        #get the absolute path of the parent folder of this file
        parent_folder_path = os.path.dirname(current_file_path)
        parent_folder_path = os.path.dirname(parent_folder_path)
        parent_folder_path = os.path.dirname(parent_folder_path)
        parent_folder_path = os.path.dirname(parent_folder_path)
        self.parent_folder_path = parent_folder_path
        
        #get setup.py file contents
        setup_path = os.path.join(parent_folder_path, "setup.py")

        #read setup_path contents
        with open(setup_path, 'r') as f:
            setup_contents = f.readlines()

        #print ("length", len(setup_contents))
        #get the list of extensions from setup.py
        extensions = []
        line_index = 0
        found = 0
        while line_index < len(setup_contents):
            line = setup_contents[line_index]
            if "CUDAExtension" in line:
                found += 1
                if found == 1:
                    continue
                #print ("extension", line, line_index)
                
                if "name"in line:
                    name_line = line.strip()
                else:
                    #get the next line
                    line_index += 1
                    name_line = setup_contents[line_index].strip()
                    
                #extract the name part
                if "name" in name_line:
                    if "'" in name_line:
                        name = name_line[name_line.find("name") + 6 : name_line.rfind("'")]
                    else:
                        name = name_line[name_line.find("name") + 6 : name_line.rfind('"')]
                    extensions.append(name)

            line_index += 1  

        return extensions


    def get_environment(self):
        """
        This method retrieves the environment for testing import
        otherwise get ImportError: libc10.so: cannot open shared object file: No such file or directory
        """
        # Get current environment and ensure CUDA/PyTorch libraries are available
        env = os.environ.copy()
        
        # Add common CUDA library paths
        ld_library_path = env.get('LD_LIBRARY_PATH', '')
        cuda_paths = [
            '/usr/local/cuda/lib64',
            '/usr/local/cuda/lib',
            '/opt/conda/lib',
            '/usr/lib/x86_64-linux-gnu'
        ]
        
        # Add PyTorch library path
        try:
            import torch
            torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.exists(torch_lib_path):
                cuda_paths.append(torch_lib_path)
        except ImportError:
            pass
        
        # Update LD_LIBRARY_PATH
        if ld_library_path:
            env['LD_LIBRARY_PATH'] = ':'.join(cuda_paths) + ':' + ld_library_path
        else:
            env['LD_LIBRARY_PATH'] = ':'.join(cuda_paths)
        return env
    

    def check_extension_import(self, extension_name, env):
        """
        Check if an extension can be imported successfully using subprocess
        Returns True if import successful, False if ImportError occurs
        """
        try:
            
            # Run Python subprocess to test the import
            result = subprocess.run([
                sys.executable, '-c', 
                'import ' + extension_name
            ], capture_output=True, text=True, timeout=30, env=env)
            print ("result.stdout", result.stdout, result.stderr)
            # Check if subprocess completed successfully
            if result.returncode != 0 and "Error" in result.stderr:
                return False, result.stderr
            else:
                return True, ""
                
        except subprocess.TimeoutExpired:
            print(f"Import test timed out for {extension_name}")
            return False, "Timeout"
        except Exception as e:
            print(f"Error testing import for {extension_name}: {e}")
            return False, str(e)


    def test_extensions_import(self):
        #get the list of extensions
        extensions = self.get_extensions_list()

        #get environment
        env = self.get_environment()

        #import all the extensions
        results = []
        for extension in extensions:
            print ("checking extension", extension)
            with self.subTest(extension=extension):
                success, error_message = self.check_extension_import(extension, env)
                #self.assertTrue(success, f"Failed to import extension: {extension}")
                results.append((extension, success, error_message))

        # Sort results by success status (True first, then False)
        sorted_results = sorted(results, key=lambda x: (not x[1], x[0]))

        #save results to a extension_import_results.txt file
        results_file_path = os.path.join(self.parent_folder_path, "extension_import_results.csv")
        with open(results_file_path, 'w') as f:
            f.write("Extension,Success,Error Message\n")
            for extension, success, error_message in results:
                f.write(f"{extension},{success},{error_message}\n")

        #print the results as a table
        print("\nExtension Import Results:")
        print("-" * 60)
        print(f"{'Extension':<30} {'Success':<10} {'Error Message':<20}")
        print("-" * 60)
        for extension, success, error_message in sorted_results:
            error_display = error_message[:17] + "..." if len(error_message) > 20 else error_message
            print(f"{extension:<30} {success:<10} {error_display:<20}")
        print("-" * 60)
        

if __name__ == '__main__':
    unittest.main()
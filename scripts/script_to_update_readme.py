import pathlib as plib

def update_readme_with_example():
    """ used to include formatted example in the README.md """
    readme_path = plib.Path(plib.Path(__file__).parent.parent, 'README.md')
    example_script_path = plib.Path(plib.Path(__file__).parent.parent,
                                    'example/example_gcms_data_analysis.py')

    # Define markers that will wrap your example content
    start_marker = '<!-- EXAMPLE_START -->'
    end_marker = '<!-- EXAMPLE_END -->'

    with open(readme_path, 'r', encoding='utf-8') as file:
        readme_content = file.read()

    # Check if the markers are present
    start_index = readme_content.find(start_marker)
    end_index = readme_content.find(end_marker)

    with open(example_script_path, 'r', encoding='utf-8') as file:
        example_content = file.read()

    formatted_example_content = f"```python\n{example_content}\n```"

    # If markers are found, replace the content between them
    if start_index != -1 and end_index != -1:
        new_readme_content = (
            readme_content[:start_index + len(start_marker)] +
            f"\n{formatted_example_content}\n" +
            readme_content[end_index:]
        )
    else:
        # If markers are not found, append the example at the end with markers
        new_readme_content = (
            readme_content +
            f"\n{start_marker}\n{formatted_example_content}\n{end_marker}\n"
        )

    with open(readme_path, 'w', encoding='utf-8') as file:
        file.write(new_readme_content)

if __name__ == "__main__":
    update_readme_with_example()

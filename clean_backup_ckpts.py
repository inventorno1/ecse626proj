import os

if __name__ == '__main__':

    backup_interval = 10
    last_backup_ckpt = 1070
    new_backup_interval = 50
    out_dir = "/cim/ehoney/ecse626proj/experiment6"
    backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')

    for epoch in range(backup_interval, last_backup_ckpt+backup_interval, backup_interval):
        if epoch % 50 != 0:
            ckpt_path = os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar')
            try:
                os.remove(ckpt_path)
                print(f"File '{ckpt_path}' has been deleted successfully.")
            except Exception as e:
                print(f"Error occurred while deleting the file: {e}")

    # for ckpt in sorted(os.listdir(backup_ckpts_dir)):
    #     print(ckpt)
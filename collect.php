<?php
use VK\Client\VKApiClient;
require "vendor/autoload.php";

function save_arr_csv($filename, $arr, $mode) {
    $file = fopen($filename, $mode);
    if ($mode == 'w') {
        fputcsv($file, array('group', 'user'));
    }
    foreach($arr as $edge) {
        fputcsv($file, $edge);
    }
    fclose($file);
}

function get_token($path) {
    $token_file = fopen("access_token", "r");
    $access_token = fread($token_file, filesize("access_token"));
    return $access_token;
}

$vk = new VKApiClient();
$access_token = get_token('access_token');

$collected_groups = array();
$collected_edges_arr = array();
save_arr_csv('data/data.csv', $collected_edges_arr, 'w');
$counter = 0;


do {
    $start = rand(1, 100000000);
    $end = $start + 10000;
    $possible_group_ids = range($start, $end, 1);
    shuffle($possible_group_ids);
    $group_ids = array_slice($possible_group_ids, 0, 500);
    $groups_response = $vk->groups()->getById($access_token,
        array(
            "group_ids" => $group_ids,
            "fields" => array("members_count")
        ));
    foreach ($groups_response as $item) {
        if (!array_key_exists($item["id"], $collected_groups) and !array_key_exists("deactivated", $item) and $item["type"] == "group" and $item["members_count"] > 100) {
            $collected_groups[$item["id"]] = 1;
            $counter += 1;
            $offset = 0;
            do {
                try {
                    usleep(333333);
                    $users_response = $vk->groups()->getMembers($access_token, array(
                        "group_id" => $item["id"],
                        "offset" => $offset
                    ));
                    $users = $users_response["items"];
                    foreach ($users as $user) {
                        array_push($collected_edges_arr, array($item["id"], $user));
                    }
                    $offset += 1000;
                }
                catch (VKApiAccessGroupException $e) {
                    echo "private group exception\n";
                    break;
                }
                catch (VKApiServerException $e1) {
                    echo "server exception\n";
                    sleep(60);
                }
                catch (VKClientException $e2) {
                    echo "client exception\n";
                    sleep(60);
                }
            } while (count($users) == 1000);
            if ($counter % 100 == 0) {
                save_arr_csv("data/data.csv", $collected_edges_arr, 'a');
                $collected_edges_arr = array();
            }
        }
    }
} while (count($collected_groups) < 30000 or count($collected_edges_arr) < 20000000);


save_arr_csv("data/data.csv", $collected_edges_arr, 'a');
?>